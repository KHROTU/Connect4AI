#define NOMINMAX
#include <iostream>
#include <vector>
#include <atomic>
#include <thread>
#include <mutex>
#include <windows.h>
#include <algorithm>
#include <chrono>
#include <random>
#include <fstream>
#include <direct.h>
#include <iomanip>
#include <Eigen/Dense>
#include <deque>

constexpr int ROWS = 6;
constexpr int COLS = 7;
constexpr int INPUT_CHANNELS = 2;
constexpr int CONV_FILTERS = 64;
constexpr int HIDDEN_SIZE = 256;
constexpr int OUTPUT_SIZE = COLS;
constexpr float LEARNING_RATE = 0.0003f;
constexpr float GAMMA = 0.99f;
constexpr int BATCH_SIZE = 2048;
constexpr int MEMORY_CAPACITY = 2000000;
constexpr int NUM_ACTORS = 8;
constexpr int NUM_LEARNERS = 2;
constexpr int TARGET_UPDATE = 1000;

using Matrix = Eigen::MatrixXf;
using Vector = Eigen::VectorXf;

struct Experience {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Eigen::Matrix<float, INPUT_CHANNELS, ROWS*COLS> state;
    int action;
    float reward;
    Eigen::Matrix<float, INPUT_CHANNELS, ROWS*COLS> next_state;
    bool done;
    float priority;
};

class ConvNet {
public:
    Eigen::Matrix<float, CONV_FILTERS, INPUT_CHANNELS*3*3> conv_weights;
    Eigen::Matrix<float, CONV_FILTERS, 1> conv_biases;
    Matrix fc1_weights;
    Vector fc1_biases;
    Matrix fc2_weights;
    Vector fc2_biases;
    Matrix fc3_weights;
    Vector fc3_biases;
    
    ConvNet() : 
        fc1_weights(HIDDEN_SIZE, CONV_FILTERS*ROWS*COLS),
        fc1_biases(HIDDEN_SIZE),
        fc2_weights(HIDDEN_SIZE, HIDDEN_SIZE),
        fc2_biases(HIDDEN_SIZE),
        fc3_weights(OUTPUT_SIZE, HIDDEN_SIZE),
        fc3_biases(OUTPUT_SIZE) {
        
        std::mt19937 gen(42);
        std::normal_distribution<float> dist(0.0f, 0.02f);
        
        auto init = [&](auto& m) { m = m.unaryExpr([&](auto x) { return dist(gen); }); };
        init(conv_weights);
        init(conv_biases);
        init(fc1_weights);
        init(fc1_biases);
        init(fc2_weights);
        init(fc2_biases);
        init(fc3_weights);
        init(fc3_biases);
    }

    Vector forward(const Eigen::Matrix<float, INPUT_CHANNELS, ROWS*COLS>& input) const {
        Eigen::Matrix<float, CONV_FILTERS, ROWS*COLS> conv_out;
        
        for(int i = 0; i < CONV_FILTERS; ++i) {
            for(int pos = 0; pos < ROWS*COLS; ++pos) {
                int r = pos / COLS;
                int c = pos % COLS;
                float sum = 0.0f;
                
                for(int ch = 0; ch < INPUT_CHANNELS; ++ch) {
                    for(int dr = -1; dr <= 1; ++dr) {
                        for(int dc = -1; dc <= 1; ++dc) {
                            int nr = r + dr;
                            int nc = c + dc;
                            if(nr >=0 && nr < ROWS && nc >=0 && nc < COLS) {
                                int input_pos = nr * COLS + nc;
                                int weight_idx = ch * 9 + (dr + 1) * 3 + (dc + 1);
                                sum += conv_weights(i, weight_idx) * input(ch, input_pos);
                            }
                        }
                    }
                }
                
                conv_out(i, pos) = std::max(sum + conv_biases[i], 0.0f);
            }
        }

        Eigen::Map<const Vector> flat_conv(conv_out.data(), conv_out.size());
        Vector fc1 = (fc1_weights * flat_conv).cwiseMax(0.0f) + fc1_biases;
        Vector fc2 = (fc2_weights * fc1).cwiseMax(0.0f) + fc2_biases;
        return fc3_weights * fc2 + fc3_biases;
    }

    void soft_update(const ConvNet& target, float tau) {
        conv_weights = tau * target.conv_weights + (1 - tau) * conv_weights;
        conv_biases = tau * target.conv_biases + (1 - tau) * conv_biases;
        fc1_weights = tau * target.fc1_weights + (1 - tau) * fc1_weights;
        fc1_biases = tau * target.fc1_biases + (1 - tau) * fc1_biases;
        fc2_weights = tau * target.fc2_weights + (1 - tau) * fc2_weights;
        fc2_biases = tau * target.fc2_biases + (1 - tau) * fc2_biases;
        fc3_weights = tau * target.fc3_weights + (1 - tau) * fc3_weights;
        fc3_biases = tau * target.fc3_biases + (1 - tau) * fc3_biases;
    }

    void save(const std::string& path) {
        std::ofstream file(path, std::ios::binary);
        file.write(reinterpret_cast<const char*>(conv_weights.data()), conv_weights.size() * sizeof(float));
        file.write(reinterpret_cast<const char*>(conv_biases.data()), conv_biases.size() * sizeof(float));
        file.write(reinterpret_cast<const char*>(fc1_weights.data()), fc1_weights.size() * sizeof(float));
        file.write(reinterpret_cast<const char*>(fc1_biases.data()), fc1_biases.size() * sizeof(float));
        file.write(reinterpret_cast<const char*>(fc2_weights.data()), fc2_weights.size() * sizeof(float));
        file.write(reinterpret_cast<const char*>(fc2_biases.data()), fc2_biases.size() * sizeof(float));
        file.write(reinterpret_cast<const char*>(fc3_weights.data()), fc3_weights.size() * sizeof(float));
        file.write(reinterpret_cast<const char*>(fc3_biases.data()), fc3_biases.size() * sizeof(float));
    }

    void load(const std::string& path) {
        std::ifstream file(path, std::ios::binary);
        file.read(reinterpret_cast<char*>(conv_weights.data()), conv_weights.size() * sizeof(float));
        file.read(reinterpret_cast<char*>(conv_biases.data()), conv_biases.size() * sizeof(float));
        file.read(reinterpret_cast<char*>(fc1_weights.data()), fc1_weights.size() * sizeof(float));
        file.read(reinterpret_cast<char*>(fc1_biases.data()), fc1_biases.size() * sizeof(float));
        file.read(reinterpret_cast<char*>(fc2_weights.data()), fc2_weights.size() * sizeof(float));
        file.read(reinterpret_cast<char*>(fc2_biases.data()), fc2_biases.size() * sizeof(float));
        file.read(reinterpret_cast<char*>(fc3_weights.data()), fc3_weights.size() * sizeof(float));
        file.read(reinterpret_cast<char*>(fc3_biases.data()), fc3_biases.size() * sizeof(float));
    }
};

class PrioritizedReplay {
private:
    std::vector<Experience, Eigen::aligned_allocator<Experience>> buffer;
    std::vector<float> priorities;
    std::atomic<size_t> position{0};
    std::mutex mtx;
    float alpha = 0.6f;

public:
    void add(const Experience& exp) {
        std::lock_guard<std::mutex> lock(mtx);
        if (buffer.size() < MEMORY_CAPACITY) {
            buffer.push_back(exp);
            priorities.push_back(exp.priority);
        } else {
            size_t pos = position % MEMORY_CAPACITY;
            buffer[pos] = exp;
            priorities[pos] = exp.priority;
        }
        position++;
    }

    std::vector<Experience, Eigen::aligned_allocator<Experience>> sample(int batch_size) {
        std::lock_guard<std::mutex> lock(mtx);
        std::vector<Experience, Eigen::aligned_allocator<Experience>> batch;
        if(buffer.empty()) return batch;

        std::vector<float> probs(priorities.begin(), priorities.end());
        for(auto& p : probs) p = std::pow(p, alpha);
        float sum = std::accumulate(probs.begin(), probs.end(), 0.0f);
        if(sum == 0.0f) return batch;
        for(auto& p : probs) p /= sum;

        std::discrete_distribution<int> dist(probs.begin(), probs.end());
        std::mt19937 gen(std::random_device{}());
        
        for(int i=0; i<batch_size; ++i) {
            int idx = dist(gen) % buffer.size();
            batch.push_back(buffer[idx]);
        }
        return batch;
    }

    void update_priorities(const std::vector<float>& new_priorities) {
        std::lock_guard<std::mutex> lock(mtx);
        for(size_t i=0; i<new_priorities.size(); ++i) {
            if(i < priorities.size()) priorities[i] = new_priorities[i];
        }
    }
};

ConvNet policy_net, target_net;
PrioritizedReplay replay_buffer;
std::atomic<bool> training_active{true};
std::atomic<int> global_step{0};
std::atomic<int> games_completed{0};
std::mutex display_mutex;

Eigen::Matrix<float, INPUT_CHANNELS, ROWS*COLS> board_to_tensor(const std::vector<std::vector<char>>& board) {
    Eigen::Matrix<float, INPUT_CHANNELS, ROWS*COLS> tensor;
    for(int r=0; r<ROWS; ++r) {
        for(int c=0; c<COLS; ++c) {
            tensor(0, r*COLS + c) = (board[r][c] == 'R') ? 1.0f : 0.0f;
            tensor(1, r*COLS + c) = (board[r][c] == 'Y') ? 1.0f : 0.0f;
        }
    }
    return tensor;
}

int select_action(const Eigen::Matrix<float, INPUT_CHANNELS, ROWS*COLS>& state, float epsilon) {
    static thread_local std::mt19937 gen(std::random_device{}());
    std::vector<int> valid_actions;
    for(int c=0; c<COLS; ++c) {
        if(state(0, 0*COLS + c) == 0 && state(1, 0*COLS + c) == 0) {
            valid_actions.push_back(c);
        }
    }
    if(valid_actions.empty()) return -1;
    
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    if(dist(gen) < epsilon) {
        return valid_actions[gen() % valid_actions.size()];
    }
    
    Vector q_values = policy_net.forward(state);
    int best_action = valid_actions[0];
    float best_value = q_values[best_action];
    for(int a : valid_actions) {
        if(q_values[a] > best_value) {
            best_action = a;
            best_value = q_values[a];
        }
    }
    return best_action;
}

bool check_win(const std::vector<std::vector<char>>& board, char player) {
    auto check = [&](int r, int c, int dr, int dc) {
        for(int i=0; i<4; ++i) {
            if(r + i*dr < 0 || r + i*dr >= ROWS || c + i*dc < 0 || c + i*dc >= COLS) return false;
            if(board[r + i*dr][c + i*dc] != player) return false;
        }
        return true;
    };
    
    for(int r=0; r<ROWS; ++r) {
        for(int c=0; c<COLS; ++c) {
            if(board[r][c] == ' ') continue;
            if((c <= COLS-4 && check(r, c, 0, 1)) ||
               (r <= ROWS-4 && check(r, c, 1, 0)) ||
               (r <= ROWS-4 && c <= COLS-4 && check(r, c, 1, 1)) ||
               (r >= 3 && c <= COLS-4 && check(r, c, -1, 1))) {
                return true;
            }
        }
    }
    return false;
}

void display_board(const std::vector<std::vector<char>>& board) {
    HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
    std::cout << "\n";
    for(int r=0; r<ROWS; ++r) {
        for(int c=0; c<COLS; ++c) {
            std::cout << "| ";
            if(board[r][c] == 'R') SetConsoleTextAttribute(hConsole, 12);
            else if(board[r][c] == 'Y') SetConsoleTextAttribute(hConsole, 14);
            std::cout << board[r][c];
            SetConsoleTextAttribute(hConsole, 7);
            std::cout << " ";
        }
        std::cout << "|\n";
    }
    std::cout << "+---+---+---+---+---+---+---+\n";
    std::cout << "  1   2   3   4   5   6   7\n";
}

void actor_thread(int thread_id, bool visualize) {
    std::vector<std::vector<char>> board(ROWS, std::vector<char>(COLS, ' '));
    float epsilon = std::pow(0.01f, static_cast<float>(thread_id)/NUM_ACTORS);
    
    while(training_active) {
        board = std::vector<std::vector<char>>(ROWS, std::vector<char>(COLS, ' '));
        char current = 'Y';
        std::vector<Experience> episode;
        bool done = false;
        
        while(!done && training_active) {
            auto state_tensor = board_to_tensor(board);
            int action = select_action(state_tensor, epsilon);
            
            if(action == -1) break;
            int row = ROWS-1;
            while(row >=0 && board[row][action] != ' ') row--;
            if(row < 0) break;
            board[row][action] = current;
            
            if(visualize) {
                std::lock_guard<std::mutex> lock(display_mutex);
                system("cls");
                display_board(board);
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
            
            float reward = 0.0f;
            if(check_win(board, current)) {
                reward = (current == 'Y') ? 1.0f : -1.0f;
                done = true;
            } else if(std::all_of(board[0].begin(), board[0].end(), [](char c) { return c != ' '; })) {
                done = true;
            }
            
            Experience exp;
            exp.state = state_tensor;
            exp.action = action;
            exp.reward = reward;
            exp.next_state = board_to_tensor(board);
            exp.done = done;
            exp.priority = 1.0f;
            
            episode.push_back(exp);
            current = (current == 'Y') ? 'R' : 'Y';
        }
        
        if(!episode.empty()) {
            for(const auto& exp : episode) replay_buffer.add(exp);
            games_completed++;
        }
    }
}

void learner_thread() {
    while(training_active) {
        auto batch = replay_buffer.sample(BATCH_SIZE);
        if(batch.empty()) continue;
        
        std::vector<float> new_priorities;
        new_priorities.reserve(batch.size());
        
        for(const auto& exp : batch) {
            Vector q_current = policy_net.forward(exp.state);
            Vector q_next = target_net.forward(exp.next_state);
            
            float target = exp.reward;
            if(!exp.done) target += GAMMA * q_next.maxCoeff();
            float td_error = std::abs(target - q_current[exp.action]);
            new_priorities.push_back(td_error + 1e-5f);
        }
        
        replay_buffer.update_priorities(new_priorities);
        
        if(global_step++ % TARGET_UPDATE == 0) {
            policy_net.soft_update(target_net, 0.01f);
        }
    }
}

void train_ai(int games) {
    _mkdir("memory");
    std::vector<std::thread> actors;
    std::vector<std::thread> learners;
    
    bool visualize = false;
    std::cout << "Visualize training? (1=Yes, 0=No): ";
    std::cin >> visualize;
    
    for(int i=0; i<NUM_ACTORS; ++i)
        actors.emplace_back(actor_thread, i, (i == 0 && visualize));
    for(int i=0; i<NUM_LEARNERS; ++i)
        learners.emplace_back(learner_thread);
    
    auto start = std::chrono::steady_clock::now();
    while(games_completed < games && training_active) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        auto now = std::chrono::steady_clock::now();
        float elapsed = std::chrono::duration<float>(now - start).count();
        std::cout << "Games: " << games_completed << "/" << games 
                  << " | Games/s: " << games_completed/elapsed << "\r";
    }
    
    training_active = false;
    for(auto& t : actors) if(t.joinable()) t.join();
    for(auto& t : learners) if(t.joinable()) t.join();
    
    policy_net.save("memory/policy.bin");
    target_net.save("memory/target.bin");
    std::cout << "\nTraining complete.\n";
}

int ai_move(const std::vector<std::vector<char>>& board) {
    auto state = board_to_tensor(board);
    int move = select_action(state, 0.0f);
    if(move == -1 || board[0][move] != ' ') {
        for(int c=0; c<COLS; ++c) {
            if(board[0][c] == ' ') {
                move = c;
                break;
            }
        }
    }
    return move;
}

int main() {
    std::cout << "1. Play\n2. Train\nChoice: ";
    int choice;
    std::cin >> choice;
    
    if(choice == 2) {
        std::cout << "Training games: ";
        int games;
        std::cin >> games;
        train_ai(games);
        return 0;
    }
    
    if(std::ifstream("memory/policy.bin")) {
        policy_net.load("memory/policy.bin");
        target_net.load("memory/target.bin");
    }
    
    std::vector<std::vector<char>> board(ROWS, std::vector<char>(COLS, ' '));
    std::cout << "Start as (1=Human, 2=AI): ";
    std::cin >> choice;
    char human = (choice == 1) ? 'R' : 'Y';
    char ai = (human == 'R') ? 'Y' : 'R';
    char current = 'R';
    
    while(true) {
        system("cls");
        display_board(board);
        
        if(current == human) {
            int move;
            do {
                std::cout << "Column (1-7): ";
                std::cin >> move;
                move--;
            } while(move < 0 || move >= COLS || board[0][move] != ' ');
            
            int row = ROWS-1;
            while(row >=0 && board[row][move] != ' ') row--;
            board[row][move] = human;
        } else {
            int move = ai_move(board);
            if(move == -1) break;
            int row = ROWS-1;
            while(row >=0 && board[row][move] != ' ') row--;
            if(row >= 0) board[row][move] = ai;
        }
        
        if(check_win(board, current)) {
            system("cls");
            display_board(board);
            std::cout << (current == human ? "Human" : "AI") << " wins!\n";
            break;
        }
        
        if(std::all_of(board[0].begin(), board[0].end(), [](char c) { return c != ' '; })) {
            system("cls");
            display_board(board);
            std::cout << "Draw!\n";
            break;
        }
        
        current = (current == 'R') ? 'Y' : 'R';
    }
    
    system("pause");
    return 0;
}