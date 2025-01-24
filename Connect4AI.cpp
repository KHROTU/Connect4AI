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
#include <unordered_map>
#include <cmath>

constexpr int ROWS = 6;
constexpr int COLS = 7;
constexpr int INPUT_CHANNELS = 2;
constexpr int CONV_FILTERS = 128;
constexpr int HIDDEN_SIZE = 512;
constexpr int OUTPUT_SIZE = COLS;
constexpr float LEARNING_RATE = 0.0002f;
constexpr float GAMMA = 0.995f;
constexpr int BATCH_SIZE = 4096;
constexpr int MEMORY_CAPACITY = 2000000;
constexpr int NUM_ACTORS = 12;
constexpr int NUM_LEARNERS = 4;
constexpr int TARGET_UPDATE = 2000;
constexpr float INITIAL_EPSILON = 0.25f;
constexpr float MIN_EPSILON = 0.02f;
constexpr int MAX_DEPTH = 8;
constexpr int MCTS_SIMS = 1500;
constexpr float C_PUCT = 4.0f;

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

struct MCTSNode {
    MCTSNode* parent;
    std::vector<std::unique_ptr<MCTSNode>> children;
    std::vector<int> valid_moves;
    int visit_count = 0;
    float value_sum = 0;
    float prior = 0;
    
    explicit MCTSNode(MCTSNode* p = nullptr) : parent(p) {}
};

class HybridNet {
public:
    Eigen::Matrix<float, CONV_FILTERS, INPUT_CHANNELS*3*3> conv_weights;
    Eigen::Matrix<float, CONV_FILTERS, 1> conv_biases;
    Matrix fc1_weights;
    Vector fc1_biases;
    Matrix fc2_weights;
    Vector fc2_biases;
    Matrix policy_head;
    Vector policy_bias;
    Matrix value_head;
    Vector value_bias;
    
    HybridNet() : 
        fc1_weights(HIDDEN_SIZE, CONV_FILTERS*ROWS*COLS),
        fc1_biases(HIDDEN_SIZE),
        fc2_weights(HIDDEN_SIZE, HIDDEN_SIZE),
        fc2_biases(HIDDEN_SIZE),
        policy_head(OUTPUT_SIZE, HIDDEN_SIZE),
        policy_bias(OUTPUT_SIZE),
        value_head(1, HIDDEN_SIZE),
        value_bias(1) {
        
        std::mt19937 gen(42);
        std::normal_distribution<float> dist(0.0f, 0.01f);
        
        auto init = [&](auto& m) { m = m.unaryExpr([&](auto x) { return dist(gen); }); };
        init(conv_weights);
        init(conv_biases);
        init(fc1_weights);
        init(fc1_biases);
        init(fc2_weights);
        init(fc2_biases);
        init(policy_head);
        init(policy_bias);
        init(value_head);
        init(value_bias);
    }

    std::pair<Vector, float> forward(const Eigen::Matrix<float, INPUT_CHANNELS, ROWS*COLS>& input) const {
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
        
        Vector policy_logits = policy_head * fc2 + policy_bias;
        float value = std::tanh((value_head * fc2 + value_bias)(0));
        return {policy_logits, value};
    }

    void soft_update(const HybridNet& target, float tau) {
        conv_weights = tau * target.conv_weights + (1 - tau) * conv_weights;
        conv_biases = tau * target.conv_biases + (1 - tau) * conv_biases;
        fc1_weights = tau * target.fc1_weights + (1 - tau) * fc1_weights;
        fc1_biases = tau * target.fc1_biases + (1 - tau) * fc1_biases;
        fc2_weights = tau * target.fc2_weights + (1 - tau) * fc2_weights;
        fc2_biases = tau * target.fc2_biases + (1 - tau) * fc2_biases;
        policy_head = tau * target.policy_head + (1 - tau) * policy_head;
        policy_bias = tau * target.policy_bias + (1 - tau) * policy_bias;
        value_head = tau * target.value_head + (1 - tau) * value_head;
        value_bias = tau * target.value_bias + (1 - tau) * value_bias;
    }

    void save(const std::string& path) {
        std::ofstream file(path, std::ios::binary);
        file.write(reinterpret_cast<const char*>(conv_weights.data()), conv_weights.size() * sizeof(float));
        file.write(reinterpret_cast<const char*>(conv_biases.data()), conv_biases.size() * sizeof(float));
        file.write(reinterpret_cast<const char*>(fc1_weights.data()), fc1_weights.size() * sizeof(float));
        file.write(reinterpret_cast<const char*>(fc1_biases.data()), fc1_biases.size() * sizeof(float));
        file.write(reinterpret_cast<const char*>(fc2_weights.data()), fc2_weights.size() * sizeof(float));
        file.write(reinterpret_cast<const char*>(fc2_biases.data()), fc2_biases.size() * sizeof(float));
        file.write(reinterpret_cast<const char*>(policy_head.data()), policy_head.size() * sizeof(float));
        file.write(reinterpret_cast<const char*>(policy_bias.data()), policy_bias.size() * sizeof(float));
        file.write(reinterpret_cast<const char*>(value_head.data()), value_head.size() * sizeof(float));
        file.write(reinterpret_cast<const char*>(value_bias.data()), value_bias.size() * sizeof(float));
    }

    void load(const std::string& path) {
        std::ifstream file(path, std::ios::binary);
        file.read(reinterpret_cast<char*>(conv_weights.data()), conv_weights.size() * sizeof(float));
        file.read(reinterpret_cast<char*>(conv_biases.data()), conv_biases.size() * sizeof(float));
        file.read(reinterpret_cast<char*>(fc1_weights.data()), fc1_weights.size() * sizeof(float));
        file.read(reinterpret_cast<char*>(fc1_biases.data()), fc1_biases.size() * sizeof(float));
        file.read(reinterpret_cast<char*>(fc2_weights.data()), fc2_weights.size() * sizeof(float));
        file.read(reinterpret_cast<char*>(fc2_biases.data()), fc2_biases.size() * sizeof(float));
        file.read(reinterpret_cast<char*>(policy_head.data()), policy_head.size() * sizeof(float));
        file.read(reinterpret_cast<char*>(policy_bias.data()), policy_bias.size() * sizeof(float));
        file.read(reinterpret_cast<char*>(value_head.data()), value_head.size() * sizeof(float));
        file.read(reinterpret_cast<char*>(value_bias.data()), value_bias.size() * sizeof(float));
    }
};

class PrioritizedReplay {
private:
    std::vector<Experience, Eigen::aligned_allocator<Experience>> buffer;
    std::vector<float> priorities;
    std::atomic<size_t> position{0};
    std::mutex mtx;
    float alpha = 0.7f;

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
        if(sum <= 0.0f) return batch;
        for(auto& p : probs) p /= sum;

        std::discrete_distribution<int> dist(probs.begin(), probs.end());
        std::mt19937 gen(std::random_device{}());
        
        for(int i=0; i<batch_size && i<buffer.size(); ++i) {
            batch.push_back(buffer[dist(gen) % buffer.size()]);
        }
        return batch;
    }

    void update_priorities(const std::vector<float>& new_priorities) {
        std::lock_guard<std::mutex> lock(mtx);
        for(size_t i=0; i<new_priorities.size() && i<priorities.size(); ++i) {
            priorities[i] = new_priorities[i];
        }
    }
};

HybridNet policy_net, target_net;
PrioritizedReplay replay_buffer;
std::atomic<bool> training_active{true};
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

std::vector<int> get_valid_moves(const std::vector<std::vector<char>>& board) {
    std::vector<int> moves;
    for(int c=0; c<COLS; ++c)
        if(board[0][c] == ' ')
            moves.push_back(c);
    return moves;
}

class MCTSSearch {
private:
    std::unique_ptr<MCTSNode> root;
    std::vector<std::vector<char>> current_board;

    float ucb_score(const MCTSNode* node, const MCTSNode* parent) const {
        if(node->visit_count == 0) return INFINITY;
        return (node->value_sum / node->visit_count) + C_PUCT * node->prior * 
               std::sqrt(parent->visit_count) / (1 + node->visit_count);
    }

    MCTSNode* select(MCTSNode* node) {
        while(!node->children.empty()) {
            auto best = std::max_element(node->children.begin(), node->children.end(),
                [&](const auto& a, const auto& b) { 
                    return ucb_score(a.get(), node) < ucb_score(b.get(), node);
                });
            node = (*best).get();
        }
        return node;
    }

    void expand(MCTSNode* node, const std::vector<int>& moves, const Vector& policy) {
        node->valid_moves = moves;
        for(int move : moves) {
            auto child = std::make_unique<MCTSNode>(node);
            child->prior = policy[move];
            node->children.push_back(std::move(child));
        }
    }

    float simulate(std::vector<std::vector<char>> board, int move) {
        int row = ROWS-1;
        while(row >=0 && board[row][move] != ' ') row--;
        board[row][move] = 'Y';
        
        auto [policy, value] = policy_net.forward(board_to_tensor(board));
        return value;
    }

    void backpropagate(MCTSNode* node, float value) {
        while(node != nullptr) {
            node->visit_count++;
            node->value_sum += value;
            value = -value;
            node = node->parent;
        }
    }

public:
    void update_root(const std::vector<std::vector<char>>& board) {
        current_board = board;
        root = std::make_unique<MCTSNode>(nullptr);
        auto [policy, value] = policy_net.forward(board_to_tensor(board));
        root->valid_moves = get_valid_moves(board);
        for(size_t i=0; i<root->valid_moves.size(); ++i) {
            auto child = std::make_unique<MCTSNode>(root.get());
            child->prior = policy[root->valid_moves[i]];
            root->children.push_back(std::move(child));
        }
    }

    void run_simulations(int num_sims) {
        for(int i=0; i<num_sims; ++i) {
            MCTSNode* node = root.get();
            std::vector<std::vector<char>> sim_board = current_board;
            
            node = select(node);
            
            if(node->visit_count > 0) {
                auto [policy, value] = policy_net.forward(board_to_tensor(sim_board));
                expand(node, get_valid_moves(sim_board), policy);
                node = node->children[0].get();
            }
            
            float value = simulate(sim_board, node->valid_moves[0]);
            backpropagate(node, value);
        }
    }

    int best_move() const {
        int best = -1;
        float best_score = -INFINITY;
        for(size_t i=0; i<root->children.size(); ++i) {
            float score = root->children[i]->visit_count;
            if(score > best_score) {
                best_score = score;
                best = root->valid_moves[i];
            }
        }
        return best;
    }
};

int neural_guided_minimax(std::vector<std::vector<char>>& board, int depth, float alpha, float beta, bool maximizing) {
    if(depth == 0) {
        auto [policy, value] = policy_net.forward(board_to_tensor(board));
        return value * (maximizing ? 1 : -1);
    }
    
    auto moves = get_valid_moves(board);
    if(moves.empty()) return 0;

    if(maximizing) {
        float max_eval = -INFINITY;
        for(int move : moves) {
            int row = ROWS-1;
            while(row >=0 && board[row][move] != ' ') row--;
            board[row][move] = 'Y';
            float eval = neural_guided_minimax(board, depth-1, alpha, beta, false);
            board[row][move] = ' ';
            max_eval = std::max(max_eval, eval);
            alpha = std::max(alpha, eval);
            if(beta <= alpha) break;
        }
        return max_eval;
    } else {
        float min_eval = INFINITY;
        for(int move : moves) {
            int row = ROWS-1;
            while(row >=0 && board[row][move] != ' ') row--;
            board[row][move] = 'R';
            float eval = neural_guided_minimax(board, depth-1, alpha, beta, true);
            board[row][move] = ' ';
            min_eval = std::min(min_eval, eval);
            beta = std::min(beta, eval);
            if(beta <= alpha) break;
        }
        return min_eval;
    }
}

int hybrid_decision(const std::vector<std::vector<char>>& board) {
    MCTSSearch mcts;
    mcts.update_root(board);
    mcts.run_simulations(MCTS_SIMS);
    int mcts_move = mcts.best_move();
    
    std::vector<std::vector<char>> temp_board = board;
    int mm_move = -1;
    float best_mm = -INFINITY;
    for(int move : get_valid_moves(board)) {
        int row = ROWS-1;
        while(row >=0 && temp_board[row][move] != ' ') row--;
        temp_board[row][move] = 'Y';
        float score = neural_guided_minimax(temp_board, 3, -INFINITY, INFINITY, false);
        temp_board[row][move] = ' ';
        if(score > best_mm) {
            best_mm = score;
            mm_move = move;
        }
    }
    
    auto [policy, value] = policy_net.forward(board_to_tensor(board));
    Vector q_values = policy;
    
    std::vector<std::pair<float, int>> candidates;
    candidates.emplace_back(q_values[mcts_move] + 0.3f, mcts_move);
    candidates.emplace_back(q_values[mm_move] + 0.2f, mm_move);
    for(int move : get_valid_moves(board))
        candidates.emplace_back(q_values[move], move);
    
    std::sort(candidates.rbegin(), candidates.rend());
    return candidates[0].second;
}

void actor_thread(int thread_id, bool visualize) {
    std::vector<std::vector<char>> board(ROWS, std::vector<char>(COLS, ' '));
    
    while(training_active) {
        float epsilon = std::max(INITIAL_EPSILON * std::pow(0.9f, games_completed/1000.0f), MIN_EPSILON);
        board = std::vector<std::vector<char>>(ROWS, std::vector<char>(COLS, ' '));
        char current = 'Y';
        std::vector<Experience> episode;
        bool done = false;
        
        while(!done && training_active) {
            auto state_tensor = board_to_tensor(board);
            int action = -1;
            
            std::mt19937 gen(std::random_device{}());
            std::uniform_real_distribution<float> dist(0.0f, 1.0f);
            if(dist(gen) < epsilon) {
                auto moves = get_valid_moves(board);
                if(!moves.empty()) action = moves[gen() % moves.size()];
            } else {
                action = hybrid_decision(board);
            }
            
            if(action == -1) break;
            int row = ROWS-1;
            while(row >=0 && board[row][action] != ' ') row--;
            if(row < 0) break;
            board[row][action] = current;

            float reward = 0.0f;
            bool win = false;
            auto check = [&](int r, int c, int dr, int dc) {
                for(int i=0; i<4; ++i) {
                    int nr = r + i*dr;
                    int nc = c + i*dc;
                    if(nr < 0 || nr >= ROWS || nc < 0 || nc >= COLS) return false;
                    if(board[nr][nc] != current) return false;
                }
                return true;
            };
            
            for(int r=0; r<ROWS; ++r) {
                for(int c=0; c<COLS; ++c) {
                    if(board[r][c] == current && 
                      (check(r,c,0,1) || check(r,c,1,0) || check(r,c,1,1) || check(r,c,-1,1))) {
                        reward = (current == 'Y') ? 1.0f : -1.0f;
                        done = true;
                        win = true;
                    }
                }
            }
            if(!win && std::all_of(board[0].begin(), board[0].end(), [](char c) { return c != ' '; })) done = true;

            if(visualize) {
                std::lock_guard<std::mutex> lock(display_mutex);
                system("cls");
                std::cout << "Training Games: " << games_completed << "\n";
                for(int r=0; r<ROWS; ++r) {
                    for(int c=0; c<COLS; ++c) 
                        std::cout << "| " << board[r][c] << " ";
                    std::cout << "|\n";
                }
                std::cout << "+---+---+---+---+---+---+---+\n";
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
            
            episode.emplace_back(Experience{state_tensor, action, reward, 
                                          board_to_tensor(board), done, 1.0f});
            current = (current == 'Y') ? 'R' : 'Y';
        }
        
        if(!episode.empty()) {
            for(const auto& exp : episode) replay_buffer.add(exp);
            games_completed++;
        }
    }
}

void learner_thread() {
    int update_counter = 0;
    while(training_active) {
        auto batch = replay_buffer.sample(BATCH_SIZE);
        if(batch.empty()) continue;
        
        std::vector<float> new_priorities;
        new_priorities.reserve(batch.size());
        
        for(const auto& exp : batch) {
            auto [q_current_policy, q_current_value] = policy_net.forward(exp.state);
            auto [q_next_policy, q_next_value] = target_net.forward(exp.next_state);
            
            float target = exp.reward;
            if(!exp.done) target += GAMMA * q_next_value;
            float td_error = std::abs(target - q_current_value);
            new_priorities.push_back(td_error + 1e-5f);
        }
        
        replay_buffer.update_priorities(new_priorities);
        
        if(++update_counter % TARGET_UPDATE == 0) {
            policy_net.soft_update(target_net, 0.01f);
        }
    }
}

void train_ai(int total_games) {
    _mkdir("memory");
    std::vector<std::thread> actors;
    std::vector<std::thread> learners;
    
    for(int i=0; i<NUM_ACTORS; ++i)
        actors.emplace_back(actor_thread, i, (i == 0));
    for(int i=0; i<NUM_LEARNERS; ++i)
        learners.emplace_back(learner_thread);
    
    auto start = std::chrono::steady_clock::now();
    while(games_completed < total_games && training_active) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        auto now = std::chrono::steady_clock::now();
        float elapsed = std::chrono::duration<float>(now - start).count();
        
        std::cout << "Progress: " << games_completed << "/" << total_games 
                  << " (" << std::fixed << std::setprecision(1) 
                  << (games_completed*100.0f/total_games) << "%)"
                  << " | Games/s: " << games_completed/elapsed << "\r";
    }
    
    training_active = false;
    for(auto& t : actors) if(t.joinable()) t.join();
    for(auto& t : learners) if(t.joinable()) t.join();
    
    policy_net.save("memory/policy.bin");
    std::cout << "\nTraining complete. Total games: " << games_completed << "\n";
}

int main() {
    std::cout << "1. Play\n2. Train\nChoice: ";
    int choice;
    std::cin >> choice;
    
    if(choice == 2) {
        std::cout << "Enter total training games: ";
        int games;
        std::cin >> games;
        train_ai(games);
        return 0;
    }
    
    if(std::ifstream("memory/policy.bin")) 
        policy_net.load("memory/policy.bin");
    
    std::vector<std::vector<char>> board(ROWS, std::vector<char>(COLS, ' '));
    char human = 'R', ai = 'Y';
    char current = 'R';
    
    while(true) {
        system("cls");
        for(int r=0; r<ROWS; ++r) {
            for(int c=0; c<COLS; ++c) 
                std::cout << "| " << board[r][c] << " ";
            std::cout << "|\n";
        }
        std::cout << "+---+---+---+---+---+---+---+\n";
        
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
            int move = hybrid_decision(board);
            int row = ROWS-1;
            while(row >=0 && board[row][move] != ' ') row--;
            board[row][move] = ai;
        }
        
        bool win = false;
        auto check = [&](int r, int c, int dr, int dc) {
            for(int i=0; i<4; ++i) {
                int nr = r + i*dr;
                int nc = c + i*dc;
                if(nr < 0 || nr >= ROWS || nc < 0 || nc >= COLS) return false;
                if(board[nr][nc] != current) return false;
            }
            return true;
        };
        
        for(int r=0; r<ROWS; ++r) {
            for(int c=0; c<COLS; ++c) {
                if(board[r][c] == current && 
                  (check(r,c,0,1) || check(r,c,1,0) || check(r,c,1,1) || check(r,c,-1,1))) {
                    system("cls");
                    for(int rr=0; rr<ROWS; ++rr) {
                        for(int cc=0; cc<COLS; ++cc) 
                            std::cout << "| " << board[rr][cc] << " ";
                        std::cout << "|\n";
                    }
                    std::cout << (current == human ? "Human" : "AI") << " wins!\n";
                    return 0;
                }
            }
        }
        
        current = (current == 'R') ? 'Y' : 'R';
    }
    
    return 0;
}