#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <random>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <map>
#include <set>
#include <queue>
#include <functional>
#include <memory>
#include <cstring>
#include <cerrno>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>

#include <torch/torch.h>

// ======================== Constants ========================
const int BATCH_SIZE = 32;
const int EPOCHS_PER_ROUND = 1;
const float THRESHOLD = 0.6f;
const int FIXED_DATA_PER_CLIENT = 5000;
const int TIMEOUT = 25;
const int R_PRIME = 100;
const int MINIMUM_ROUNDS = 40;
const int COUNT_THRESHOLD = 5;
const int BASE_PORT = 8650;
const int IMAGE_SIZE = 32;
const int NUM_CLASSES = 10;
const int NUM_CHANNELS = 3;
const int MAX_MESSAGE_SIZE = 100 * 1024 * 1024; // 100MB

// ======================== Global Variables ========================
int NUM_CLIENTS = 0;
int NUM_MACHINES = 0;
std::string CURRENT_MACHINE_IP;
std::vector<std::string> ips;
std::vector<std::tuple<int, int, int>> faults;
std::vector<std::vector<int>> adj;
std::atomic<int> total_model_messages{0};
std::atomic<int> total_terminate_messages{0};
std::mutex log_mutex;
std::ofstream log_file;

// ======================== RAII Socket Wrapper ========================
class SocketRAII {
private:
    int sock_;
public:
    explicit SocketRAII(int sock = -1) : sock_(sock) {}
    ~SocketRAII() { 
        if (sock_ >= 0) {
            shutdown(sock_, SHUT_RDWR);
            close(sock_); 
        }
    }
    int get() const { return sock_; }
    void set(int sock) { sock_ = sock; }
    int release() { 
        int temp = sock_; 
        sock_ = -1; 
        return temp; 
    }
    bool valid() const { return sock_ >= 0; }
    
    SocketRAII(const SocketRAII&) = delete;
    SocketRAII& operator=(const SocketRAII&) = delete;
    SocketRAII(SocketRAII&& other) noexcept : sock_(other.sock_) {
        other.sock_ = -1;
    }
    SocketRAII& operator=(SocketRAII&& other) noexcept {
        if (this != &other) {
            if (sock_ >= 0) close(sock_);
            sock_ = other.sock_;
            other.sock_ = -1;
        }
        return *this;
    }
};

// ======================== Thread Pool ========================
class ThreadPool {
private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queue_mutex;
    std::condition_variable condition;
    std::atomic<bool> stop{false};
    
public:
    explicit ThreadPool(size_t threads) {
        for(size_t i = 0; i < threads; ++i) {
            workers.emplace_back([this] {
                while(true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(this->queue_mutex);
                        this->condition.wait(lock, [this]{ 
                            return this->stop.load() || !this->tasks.empty(); 
                        });
                        if(this->stop.load() && this->tasks.empty()) return;
                        task = std::move(this->tasks.front());
                        this->tasks.pop();
                    }
                    try {
                        task();
                    } catch (const std::exception& e) {
                        std::cerr << "Task exception: " << e.what() << std::endl;
                    }
                }
            });
        }
    }
    
    template<class F>
    void enqueue(F&& f) {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            if (stop.load()) {
                throw std::runtime_error("enqueue on stopped ThreadPool");
            }
            tasks.emplace(std::forward<F>(f));
        }
        condition.notify_one();
    }
    
    void wait_all() {
        while(true) {
            std::unique_lock<std::mutex> lock(queue_mutex);
            if (tasks.empty()) break;
            lock.unlock();
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
    
    ~ThreadPool() {
        stop.store(true);
        condition.notify_all();
        for(std::thread &worker: workers) {
            if (worker.joinable()) worker.join();
        }
    }
};

// ======================== Logging ========================
void log_message(const std::string& message) {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::string time_str = std::ctime(&time);
    time_str.pop_back();
    
    std::lock_guard<std::mutex> lock(log_mutex);
    std::cout << time_str << " - INFO - " << message << std::endl;
    
    std::string lower_msg = message;
    std::transform(lower_msg.begin(), lower_msg.end(), lower_msg.begin(), ::tolower);
    if (lower_msg.find("crash") != std::string::npos) {
        log_file << time_str << " - INFO - " << message << std::endl;
        log_file.flush();
    }
}

// ======================== CIFAR-10 Dataset ========================
class CIFAR10Dataset : public torch::data::Dataset<CIFAR10Dataset> {
private:
    torch::Tensor images_;
    torch::Tensor labels_;

public:
    CIFAR10Dataset(const std::string& root, bool train, const std::vector<size_t>& indices = {}) {
        load_cifar10(root, train);
        
        if (!indices.empty()) {
            torch::Tensor subset_images = torch::empty({static_cast<int64_t>(indices.size()), 3, 32, 32});
            torch::Tensor subset_labels = torch::empty({static_cast<int64_t>(indices.size())}, torch::kInt64);
            
            for (size_t i = 0; i < indices.size(); i++) {
                subset_images[i] = images_[indices[i]];
                subset_labels[i] = labels_[indices[i]];
            }
            
            images_ = subset_images;
            labels_ = subset_labels;
        }
    }

    torch::data::Example<> get(size_t index) override {
        return {images_[index].clone(), labels_[index].clone()};
    }

    torch::optional<size_t> size() const override {
        return images_.size(0);
    }

private:
    void load_cifar10(const std::string& root, bool train) {
        std::string data_dir = root + "/cifar-10-batches-bin/";
        std::vector<std::string> files;
        
        if (train) {
            for (int i = 1; i <= 5; i++) {
                files.push_back(data_dir + "data_batch_" + std::to_string(i) + ".bin");
            }
        } else {
            files.push_back(data_dir + "test_batch.bin");
        }
        
        std::vector<torch::Tensor> all_images;
        std::vector<torch::Tensor> all_labels;
        
        for (const auto& file : files) {
            std::ifstream ifs(file, std::ios::binary);
            if (!ifs.is_open()) {
                std::cerr << "Warning: Could not open " << file << ". Creating dummy data." << std::endl;
                create_dummy_data(train);
                return;
            }
            
            const int num_images = 10000;
            const int image_bytes = 3072;
            
            for (int i = 0; i < num_images; i++) {
                uint8_t label;
                ifs.read(reinterpret_cast<char*>(&label), 1);
                
                std::vector<uint8_t> image_data(image_bytes);
                ifs.read(reinterpret_cast<char*>(image_data.data()), image_bytes);
                
                torch::Tensor img = torch::from_blob(image_data.data(), {3, 32, 32}, torch::kUInt8).clone();
                img = img.to(torch::kFloat32) / 255.0f;
                img = (img - 0.5f) / 0.5f;
                
                all_images.push_back(img);
                all_labels.push_back(torch::tensor(static_cast<int64_t>(label)));
            }
            
            ifs.close();
        }
        
        if (!all_images.empty()) {
            images_ = torch::stack(all_images);
            labels_ = torch::stack(all_labels);
        } else {
            create_dummy_data(train);
        }
    }
    
    void create_dummy_data(bool train) {
        int num_samples = train ? 50000 : 10000;
        images_ = torch::randn({num_samples, 3, 32, 32});
        labels_ = torch::randint(0, 10, {num_samples}, torch::kInt64);
    }
};

// ======================== SimpleCNN Model ========================
class SimpleCNNImpl : public torch::nn::Module {
public:
    SimpleCNNImpl() {
        conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 32, 3)));
        conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 3)));
        fc1 = register_module("fc1", torch::nn::Linear(64 * 6 * 6, 128));
        fc2 = register_module("fc2", torch::nn::Linear(128, 10));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(conv1->forward(x));
        x = torch::max_pool2d(x, 2);
        x = torch::relu(conv2->forward(x));
        x = torch::max_pool2d(x, 2);
        x = x.view({x.size(0), -1});
        x = torch::relu(fc1->forward(x));
        x = fc2->forward(x);
        return x;
    }

    torch::nn::Conv2d conv1{nullptr};
    torch::nn::Conv2d conv2{nullptr};
    torch::nn::Linear fc1{nullptr};
    torch::nn::Linear fc2{nullptr};
};

TORCH_MODULE(SimpleCNN);

// ======================== Message Structure ========================
struct Message {
    std::string type;
    std::vector<uint8_t> weights_data;
    int round;
    int terminate;
    int id;
};

// ======================== Binary Serialization ========================
std::string serialize_message(const Message& msg) {
    std::ostringstream oss(std::ios::binary);
    
    // Write type
    uint32_t type_len = msg.type.length();
    oss.write(reinterpret_cast<const char*>(&type_len), sizeof(type_len));
    oss.write(msg.type.c_str(), type_len);
    
    // Write metadata
    oss.write(reinterpret_cast<const char*>(&msg.round), sizeof(msg.round));
    oss.write(reinterpret_cast<const char*>(&msg.terminate), sizeof(msg.terminate));
    oss.write(reinterpret_cast<const char*>(&msg.id), sizeof(msg.id));
    
    // Write weights data
    uint64_t data_size = msg.weights_data.size();
    oss.write(reinterpret_cast<const char*>(&data_size), sizeof(data_size));
    if (data_size > 0) {
        oss.write(reinterpret_cast<const char*>(msg.weights_data.data()), data_size);
    }
    
    return oss.str();
}

Message deserialize_message(const std::string& data) {
    Message msg;
    std::istringstream iss(data, std::ios::binary);
    
    // Read type
    uint32_t type_len;
    iss.read(reinterpret_cast<char*>(&type_len), sizeof(type_len));
    msg.type.resize(type_len);
    iss.read(&msg.type[0], type_len);
    
    // Read metadata
    iss.read(reinterpret_cast<char*>(&msg.round), sizeof(msg.round));
    iss.read(reinterpret_cast<char*>(&msg.terminate), sizeof(msg.terminate));
    iss.read(reinterpret_cast<char*>(&msg.id), sizeof(msg.id));
    
    // Read weights data
    uint64_t data_size;
    iss.read(reinterpret_cast<char*>(&data_size), sizeof(data_size));
    if (data_size > 0) {
        msg.weights_data.resize(data_size);
        iss.read(reinterpret_cast<char*>(msg.weights_data.data()), data_size);
    }
    
    return msg;
}

// ======================== Model Serialization ========================
std::vector<uint8_t> serialize_weights(const SimpleCNN& model) {
    std::ostringstream oss(std::ios::binary);
    torch::save(model, oss);
    std::string str = oss.str();
    return std::vector<uint8_t>(str.begin(), str.end());
}

void deserialize_weights(SimpleCNN& model, const std::vector<uint8_t>& data) {
    std::string str(data.begin(), data.end());
    std::istringstream iss(str, std::ios::binary);
    torch::load(model, iss);
}

// ======================== Network Operations ========================
bool send_all(int sock, const void* data, size_t length) {
    size_t total_sent = 0;
    const char* ptr = static_cast<const char*>(data);
    
    while (total_sent < length) {
        ssize_t sent = send(sock, ptr + total_sent, length - total_sent, MSG_NOSIGNAL);
        if (sent < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        if (sent == 0) return false;
        total_sent += sent;
    }
    return true;
}

bool recv_all(int sock, void* data, size_t length) {
    size_t total_received = 0;
    char* ptr = static_cast<char*>(data);
    
    while (total_received < length) {
        ssize_t received = recv(sock, ptr + total_received, length - total_received, MSG_WAITALL);
        if (received < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        if (received == 0) return false;
        total_received += received;
    }
    return true;
}

bool send_message(int sock, const Message& message) {
    try {
        std::string data = serialize_message(message);
        uint32_t length = htonl(static_cast<uint32_t>(data.size()));
        
        if (!send_all(sock, &length, sizeof(length))) return false;
        if (!send_all(sock, data.c_str(), data.size())) return false;
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Send error: " << e.what() << std::endl;
        return false;
    }
}

Message receive_message(int sock) {
    uint32_t length;
    if (!recv_all(sock, &length, sizeof(length))) {
        throw std::runtime_error("Failed to receive message length");
    }
    length = ntohl(length);
    
    if (length == 0 || length > MAX_MESSAGE_SIZE) {
        throw std::runtime_error("Invalid message size: " + std::to_string(length));
    }
    
    std::vector<char> buffer(length);
    if (!recv_all(sock, buffer.data(), length)) {
        throw std::runtime_error("Failed to receive message data");
    }
    
    std::string data(buffer.begin(), buffer.end());
    return deserialize_message(data);
}

// ======================== TCP Client ========================
void tcp_client(int id, int target_id, const std::string& target_ip, const Message& message) {
    SocketRAII sock(socket(AF_INET, SOCK_STREAM, 0));
    if (!sock.valid()) {
        std::cerr << "Failed to create socket: " << strerror(errno) << std::endl;
        return;
    }
    
    // Set timeouts
    struct timeval timeout;
    timeout.tv_sec = 5;
    timeout.tv_usec = 0;
    setsockopt(sock.get(), SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    setsockopt(sock.get(), SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
    
    // Enable TCP_NODELAY for lower latency
    int flag = 1;
    setsockopt(sock.get(), IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));
    
    struct sockaddr_in serv_addr;
    memset(&serv_addr, 0, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(BASE_PORT + target_id);
    
    if (inet_pton(AF_INET, target_ip.c_str(), &serv_addr.sin_addr) <= 0) {
        std::cerr << "Invalid address: " << target_ip << std::endl;
        return;
    }
    
    int retries = 1;
    while (retries > 0) {
        if (connect(sock.get(), (struct sockaddr*)&serv_addr, sizeof(serv_addr)) == 0) {
            if (send_message(sock.get(), message)) {
                break;
            }
        }
        retries--;
        if (retries > 0) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }
}

// ======================== Broadcast Functions ========================
void broadcast_weights(int id, const SimpleCNN& model, int current_round, int terminate,
                      ThreadPool& pool) {
    Message message;
    message.type = "weights";
    message.weights_data = serialize_weights(model);
    message.round = current_round;
    message.terminate = terminate;
    message.id = id;
    
    for (int pid : adj[id]) {
        total_model_messages++;
        pool.enqueue([id, pid, ip = ips[pid], message]() {
            tcp_client(id, pid, ip, message);
        });
    }
}

void broadcast_terminate(int id, ThreadPool& pool) {
    Message message;
    message.type = "terminate";
    message.round = 0;
    message.terminate = 0;
    message.id = id;
    
    for (int pid : adj[id]) {
        total_terminate_messages++;
        pool.enqueue([id, pid, ip = ips[pid], message]() {
            tcp_client(id, pid, ip, message);
        });
    }
}

// ======================== TCP Server ========================
void tcp_server(int id, std::vector<Message>& received_weights, 
               std::atomic<bool>& terminate_flag,
               const std::string& local_ip, 
               std::map<int, std::shared_ptr<SimpleCNNImpl>>& latest_models,
               std::mutex& received_mutex) {
    SocketRAII server_fd(socket(AF_INET, SOCK_STREAM, 0));
    if (!server_fd.valid()) {
        std::cerr << "Failed to create server socket for client " << id << std::endl;
        return;
    }
    
    int opt = 1;
    setsockopt(server_fd.get(), SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    setsockopt(server_fd.get(), SOL_SOCKET, SO_REUSEPORT, &opt, sizeof(opt));
    
    struct sockaddr_in address;
    memset(&address, 0, sizeof(address));
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = inet_addr(local_ip.c_str());
    address.sin_port = htons(BASE_PORT + id);
    
    if (bind(server_fd.get(), (struct sockaddr*)&address, sizeof(address)) < 0) {
        std::cerr << "Bind failed for client " << id << ": " << strerror(errno) << std::endl;
        return;
    }
    
    if (listen(server_fd.get(), NUM_CLIENTS) < 0) {
        std::cerr << "Listen failed for client " << id << std::endl;
        return;
    }
    
    while (!terminate_flag.load()) {
        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);
        
        // Set timeout for accept
        struct timeval timeout;
        timeout.tv_sec = 1;
        timeout.tv_usec = 0;
        setsockopt(server_fd.get(), SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
        
        int client_sock = accept(server_fd.get(), (struct sockaddr*)&client_addr, &client_len);
        
        if (client_sock < 0) {
            if (errno == EWOULDBLOCK || errno == EAGAIN) {
                continue;
            }
            continue;
        }
        
        SocketRAII client(client_sock);
        
        try {
            Message msg = receive_message(client.get());
            
            std::lock_guard<std::mutex> lock(received_mutex);
            
            if (msg.type == "terminate") {
                terminate_flag.store(true);
                break;
            }
            
            if (msg.type == "weights") {
                received_weights.push_back(msg);
                if (msg.terminate == 1) {
                    terminate_flag.store(true);
                }
                
                auto temp_model = std::make_shared<SimpleCNNImpl>();
                SimpleCNN temp_wrapper(temp_model);
                deserialize_weights(temp_wrapper, msg.weights_data);
                latest_models[msg.id] = temp_model;
            }
        } catch (const std::exception& e) {
            // Connection error, continue
        }
    }
}

// ======================== Model Operations ========================
void average_models(SimpleCNN& target_model, const std::vector<SimpleCNN>& models) {
    if (models.empty()) return;
    
    auto target_params = target_model->named_parameters();
    
    for (auto& target_pair : target_params) {
        torch::Tensor sum = torch::zeros_like(target_pair.value());
        
        for (const auto& model : models) {
            auto params = model->named_parameters();
            // FIX: Use find() which returns an iterator, then check and dereference
            for (const auto& param_pair : params) {
                if (param_pair.key() == target_pair.key()) {
                    sum = sum + param_pair.value().detach().clone();
                    break;
                }
            }
        }
        
        target_pair.value().data().copy_(sum / static_cast<float>(models.size()));
    }
}

bool models_are_similar(const SimpleCNN& model1, const SimpleCNN& model2, float threshold) {
    auto params1 = model1->named_parameters();
    auto params2 = model2->named_parameters();
    
    for (const auto& pair1 : params1) {
        // FIX: Iterate through params2 to find matching key
        bool found = false;
        for (const auto& pair2 : params2) {
            if (pair2.key() == pair1.key()) {
                torch::Tensor diff = pair1.value() - pair2.value();
                float norm = torch::norm(diff).item<float>();
                if (norm > threshold) {
                    return false;
                }
                found = true;
                break;
            }
        }
        if (!found) return false;
    }
    return true;
}

// FIX: Change parameter type to accept the correct data loader type
template<typename DataLoader>
float compute_accuracy(SimpleCNN& model, DataLoader& data_loader) {
    model->eval();
    torch::NoGradGuard no_grad;
    
    int64_t correct = 0;
    int64_t total = 0;
    
    for (auto& batch : data_loader) {
        auto data = batch.data;
        auto target = batch.target;
        
        auto output = model->forward(data);
        auto pred = output.argmax(1);
        
        correct += pred.eq(target).sum().template item<int64_t>();
        total += target.size(0);
    }
    
    return 100.0f * correct / total;
}

// ======================== Data Splitting ========================
std::vector<std::vector<size_t>> create_dirichlet_splits(int num_samples, int num_clients, float alpha) {
    std::vector<std::vector<size_t>> client_indices(num_clients);
    std::vector<size_t> all_indices(num_samples);
    std::iota(all_indices.begin(), all_indices.end(), 0);
    
    std::mt19937 g(42);
    std::shuffle(all_indices.begin(), all_indices.end(), g);
    
    size_t idx = 0;
    for (int i = 0; i < num_clients; i++) {
        size_t client_size = FIXED_DATA_PER_CLIENT;
        for (size_t j = 0; j < client_size && idx < num_samples; j++, idx++) {
            client_indices[i].push_back(all_indices[idx % num_samples]);
        }
    }
    
    return client_indices;
}

// ======================== Input Parsing ========================
bool parse_input_file() {
    std::ifstream file("inputf.txt");
    if (!file.is_open()) {
        std::cerr << "Failed to open inputf.txt" << std::endl;
        return false;
    }

    std::string line;
    
    std::getline(file, line);
    std::istringstream iss1(line);
    iss1 >> NUM_CLIENTS >> NUM_MACHINES;
    
    std::getline(file, CURRENT_MACHINE_IP);
    
    std::getline(file, line);
    std::istringstream iss2(line);
    std::string ip;
    while (std::getline(iss2, ip, ',')) {
        ips.push_back(ip);
    }
    
    int num_faults;
    std::getline(file, line);
    num_faults = std::stoi(line);
    
    for (int i = 0; i < num_faults; i++) {
        std::getline(file, line);
        std::istringstream iss3(line);
        int id, fr, y;
        char comma;
        iss3 >> id >> comma >> fr >> comma >> y;
        faults.push_back(std::make_tuple(id, fr, y));
    }
    
    file.close();
    
    adj.resize(NUM_CLIENTS);
    for (int i = 0; i < NUM_CLIENTS; i++) {
        for (int j = 0; j < NUM_CLIENTS; j++) {
            if (i != j) {
                adj[i].push_back(j);
            }
        }
    }
    
    return true;
}

// ======================== Client Logic ========================
void client_logic(int id, const std::string& local_ip, 
                 const std::vector<size_t>& train_indices,
                 const std::string& data_root) {
    log_message("Client " + std::to_string(id) + " started");
    
    torch::manual_seed(42);
    
    // Create thread pool for this client
    ThreadPool pool(std::min(8, NUM_CLIENTS));
    
    // Load datasets
    auto train_dataset = CIFAR10Dataset(data_root, true, train_indices)
        .map(torch::data::transforms::Stack<>());
    auto train_loader = torch::data::make_data_loader(
        std::move(train_dataset),
        torch::data::DataLoaderOptions().batch_size(BATCH_SIZE).workers(0)
    );
    
    auto test_dataset = CIFAR10Dataset(data_root, false)
        .map(torch::data::transforms::Stack<>());
    auto test_loader = torch::data::make_data_loader(
        std::move(test_dataset),
        torch::data::DataLoaderOptions().batch_size(BATCH_SIZE).workers(0)
    );
    
    SimpleCNN model;
    SimpleCNN previous_model;
    bool has_previous = false;
    
    torch::optim::SGD optimizer(model->parameters(), torch::optim::SGDOptions(0.01).momentum(0.9));
    
    int current_round = 0;
    std::vector<Message> received_weights;
    std::atomic<bool> terminate_flag{false};
    std::mutex received_mutex;
    int counter = 0;
    std::map<int, std::shared_ptr<SimpleCNNImpl>> latest_models;
    std::vector<bool> crash_away_list(NUM_CLIENTS, false);
    std::set<int> crashed_in_rounds;
    
    std::thread server_thread(tcp_server, id, std::ref(received_weights), 
                            std::ref(terminate_flag), local_ip, 
                            std::ref(latest_models), std::ref(received_mutex));
    
    std::this_thread::sleep_for(std::chrono::seconds(2));
    
    while (current_round < R_PRIME && !terminate_flag.load()) {
        model->train();
        
        // Training
        for (int epoch = 0; epoch < EPOCHS_PER_ROUND; epoch++) {
            for (auto& batch : *train_loader) {
                auto data = batch.data;
                auto target = batch.target;
                
                optimizer.zero_grad();
                auto output = model->forward(data);
                auto loss = torch::nn::functional::cross_entropy(output, target);
                loss.backward();
                optimizer.step();
            }
        }
        
        // Check for crash
        for (const auto& fault : faults) {
            if (std::get<0>(fault) == id && std::get<1>(fault) == current_round) {
                for (int i = 0; i < std::get<2>(fault); i++) {
                    broadcast_weights(id, model, current_round, 0, pool);
                }
                pool.wait_all();
                log_message("Client " + std::to_string(id) + " is crashing at round " + std::to_string(current_round));
                terminate_flag.store(true);
                if (server_thread.joinable()) server_thread.join();
                return;
            }
        }
        
        // Check termination flag
        if (terminate_flag.load()) {
            log_message("Client " + std::to_string(id) + " received termination flag at round " + std::to_string(current_round));
            broadcast_weights(id, model, current_round, 1, pool);
            pool.wait_all();
            break;
        }
        
        broadcast_weights(id, model, current_round, 0, pool);
        pool.wait_all();
        
        auto start_time = std::chrono::steady_clock::now();
        while (std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now() - start_time).count() < TIMEOUT) {
            if (terminate_flag.load()) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        
        std::vector<Message> received_copy;
        {
            std::lock_guard<std::mutex> lock(received_mutex);
            received_copy = received_weights;
        }
        
        // Detect crashes
        bool new_crashes = false;
        std::set<int> received_ids;
        for (const auto& msg : received_copy) {
            received_ids.insert(msg.id);
        }
        
        for (int client_id = 0; client_id < NUM_CLIENTS; client_id++) {
            if (client_id != id && received_ids.find(client_id) == received_ids.end() &&
                !crash_away_list[client_id]) {
                crash_away_list[client_id] = true;
                new_crashes = true;
                log_message("Client " + std::to_string(id) + " detected crash of client " +
                          std::to_string(client_id) + " at round " + std::to_string(current_round));
            }
        }
        
        if (new_crashes) {
            crashed_in_rounds.insert(current_round);
        }
        
        // Average models
        std::vector<SimpleCNN> all_models;
        for (const auto& msg : received_copy) {
            SimpleCNN temp_model;
            deserialize_weights(temp_model, msg.weights_data);
            all_models.push_back(temp_model);
        }
        all_models.push_back(model);
        
        SimpleCNN new_model;
        average_models(new_model, all_models);
        
        auto new_params = new_model->named_parameters();
        auto model_params = model->named_parameters();
        for (auto& pair : model_params) {
            // FIX: Find matching parameter in new_params
            for (const auto& new_pair : new_params) {
                if (new_pair.key() == pair.key()) {
                    pair.value().data().copy_(new_pair.value());
                    break;
                }
            }
        }
        
        // FIX: Pass the data loader by reference with dereference operator
        float accuracy = compute_accuracy(model, *test_loader);
        log_message("Client " + std::to_string(id) + " - Round " + std::to_string(current_round) +
                   ": Accuracy: " + std::to_string(accuracy) + "%");
        
        if (current_round >= MINIMUM_ROUNDS) {
            if (has_previous && models_are_similar(new_model, previous_model, THRESHOLD)) {
                counter++;
            } else {
                counter = 0;
            }
            
            bool no_recent_crashes = true;
            for (int r = current_round - COUNT_THRESHOLD + 1; r <= current_round; r++) {
                if (crashed_in_rounds.find(r) != crashed_in_rounds.end()) {
                    no_recent_crashes = false;
                    break;
                }
            }
            
            if (counter >= COUNT_THRESHOLD && no_recent_crashes) {
                log_message("Client " + std::to_string(id) + " met termination criteria at round " +
                          std::to_string(current_round));
                broadcast_weights(id, model, current_round, 1, pool);
                pool.wait_all();
                break;
            }
        }
        
        previous_model = new_model;
        has_previous = true;
        current_round++;
        
        {
            std::lock_guard<std::mutex> lock(received_mutex);
            received_weights.clear();
        }
    }
    
    if (current_round == R_PRIME) {
        log_message("Client " + std::to_string(id) + " reached maximum rounds");
    }
    
    broadcast_terminate(id, pool);
    pool.wait_all();
    
    terminate_flag.store(true);
    if (server_thread.joinable()) {
        server_thread.join();
    }
}

// ======================== Main Function ========================
int main(int argc, char* argv[]) {
    if (!parse_input_file()) {
        std::cerr << "Failed to parse input file" << std::endl;
        return 1;
    }
    
    std::string log_filename = "min40_crash_test_" + std::to_string(TIMEOUT) + "_log_" +
                              std::to_string(NUM_CLIENTS) + "_" + std::to_string(NUM_MACHINES) +
                              "_" + std::to_string(faults.size()) + ".txt";
    log_file.open(log_filename, std::ios::out | std::ios::trunc);
    if (!log_file.is_open()) {
        std::cerr << "Failed to open log file: " << log_filename << std::endl;
        return 1;
    }
    
    auto start_time = std::chrono::steady_clock::now();
    log_message("Starting Federated Learning on Machine " + CURRENT_MACHINE_IP);
    log_message("Number of Clients: " + std::to_string(NUM_CLIENTS));
    log_message("Number of Machines: " + std::to_string(NUM_MACHINES));
    log_message("Number of Faults: " + std::to_string(faults.size()));
    
    std::string data_root = "./data";
    
    // Create data splits
    auto client_indices = create_dirichlet_splits(50000, NUM_CLIENTS, 0.5);
    
    std::vector<std::thread> threads;
    for (int i = 0; i < NUM_CLIENTS; i++) {
        if (ips[i] == CURRENT_MACHINE_IP) {
            threads.emplace_back(client_logic, i, CURRENT_MACHINE_IP, client_indices[i], data_root);
        }
    }
    
    for (auto& t : threads) {
        if (t.joinable()) {
            t.join();
        }
    }
    
    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();
    
    int total_model_msgs = total_model_messages.load();
    int total_term_msgs = total_terminate_messages.load();
    
    log_message("\n========================================");
    log_message("Federated Learning Completed");
    log_message("========================================");
    log_message("Current Machine IP: " + CURRENT_MACHINE_IP);
    log_message("Number of Clients: " + std::to_string(NUM_CLIENTS));
    log_message("Total model messages passed: " + 
                std::to_string(total_model_msgs - ((NUM_CLIENTS/2)*(NUM_CLIENTS-1)) - total_term_msgs));
    log_message("Total Termination Messages Passed: " + std::to_string(total_term_msgs));
    log_message("Total Time Taken: " + std::to_string(duration) + " seconds");
    log_message("========================================");
    
    log_file.close();
    return 0;
}