#define NOMINMAX
#include <iostream>
#include <vector>
#include <unordered_map>
#include <windows.h>
#include <limits>
#include <algorithm>
#include <chrono>
#include <random>
#include <bitset>
#include <fstream>
#include <direct.h>
#include <numeric>
#include <iomanip>

const int ROWS = 6;
const int COLS = 7;
const int INPUT_SIZE = ROWS * COLS * 2;
const int HIDDEN_SIZE = 256;
const int OUTPUT_SIZE = COLS;
const double LEARNING_RATE = 0.0005;
const double GAMMA = 0.97;
const int BATCH_SIZE = 256;
const int MEMORY_CAPACITY = 1000000;
const int TARGET_UPDATE = 50;
const double EPS_START = 1.0;
const double EPS_END = 0.01;
const double EPS_DECAY = 0.9997;
const double TAU = 0.005;

struct Experience {
    std::bitset<INPUT_SIZE> state;
    int action;
    double reward;
    std::bitset<INPUT_SIZE> next_state;
    bool done;
};

struct NeuralNetwork {
    double W1[INPUT_SIZE][HIDDEN_SIZE];
    double b1[HIDDEN_SIZE];
    double W2[HIDDEN_SIZE][HIDDEN_SIZE];
    double b2[HIDDEN_SIZE];
    double W3[HIDDEN_SIZE][OUTPUT_SIZE];
    double b3[OUTPUT_SIZE];
    
    NeuralNetwork() {
        std::mt19937 gen(42);
        std::normal_distribution<double> dist(0.0, 0.1);
        
        auto init_weights = [&](auto& arr, int size1, int size2) {
            for(int i=0; i<size1; ++i)
                for(int j=0; j<size2; ++j)
                    arr[i][j] = dist(gen);
        };
        
        init_weights(W1, INPUT_SIZE, HIDDEN_SIZE);
        init_weights(W2, HIDDEN_SIZE, HIDDEN_SIZE);
        init_weights(W3, HIDDEN_SIZE, OUTPUT_SIZE);
        std::fill(b1, b1+HIDDEN_SIZE, 0.0);
        std::fill(b2, b2+HIDDEN_SIZE, 0.0);
        std::fill(b3, b3+OUTPUT_SIZE, 0.0);
    }

    double relu(double x) const { return x > 0 ? x : 0; }

    std::vector<double> forward(const std::bitset<INPUT_SIZE>& input) const {
        double h1[HIDDEN_SIZE] = {0};
        for(int j=0; j<HIDDEN_SIZE; ++j) {
            for(int i=0; i<INPUT_SIZE; ++i)
                h1[j] += input[i] * W1[i][j];
            h1[j] = relu(h1[j] + b1[j]);
        }
        
        double h2[HIDDEN_SIZE] = {0};
        for(int j=0; j<HIDDEN_SIZE; ++j) {
            for(int i=0; i<HIDDEN_SIZE; ++i)
                h2[j] += h1[i] * W2[i][j];
            h2[j] = relu(h2[j] + b2[j]);
        }
        
        std::vector<double> output(OUTPUT_SIZE);
        for(int j=0; j<OUTPUT_SIZE; ++j) {
            for(int i=0; i<HIDDEN_SIZE; ++i)
                output[j] += h2[i] * W3[i][j];
            output[j] += b3[j];
        }
        return output;
    }

    void soft_update(const NeuralNetwork& target) {
        for(int i=0; i<INPUT_SIZE; ++i)
            for(int j=0; j<HIDDEN_SIZE; ++j)
                W1[i][j] = TAU * target.W1[i][j] + (1-TAU) * W1[i][j];
        
        for(int i=0; i<HIDDEN_SIZE; ++i) {
            b1[i] = TAU * target.b1[i] + (1-TAU) * b1[i];
            for(int j=0; j<HIDDEN_SIZE; ++j)
                W2[i][j] = TAU * target.W2[i][j] + (1-TAU) * W2[i][j];
            b2[i] = TAU * target.b2[i] + (1-TAU) * b2[i];
        }
        
        for(int i=0; i<HIDDEN_SIZE; ++i)
            for(int j=0; j<OUTPUT_SIZE; ++j)
                W3[i][j] = TAU * target.W3[i][j] + (1-TAU) * W3[i][j];
        
        for(int j=0; j<OUTPUT_SIZE; ++j)
            b3[j] = TAU * target.b3[j] + (1-TAU) * b3[j];
    }

    void update(const NeuralNetwork& target, const std::vector<Experience>& batch) {
        double dW1[INPUT_SIZE][HIDDEN_SIZE] = {0};
        double db1[HIDDEN_SIZE] = {0};
        double dW2[HIDDEN_SIZE][HIDDEN_SIZE] = {0};
        double db2[HIDDEN_SIZE] = {0};
        double dW3[HIDDEN_SIZE][OUTPUT_SIZE] = {0};
        double db3[OUTPUT_SIZE] = {0};

        for(const auto& exp : batch) {
            auto q_current = forward(exp.state);
            auto q_next = target.forward(exp.next_state);
            
            double target_val = exp.reward;
            if(!exp.done)
                target_val += GAMMA * *std::max_element(q_next.begin(), q_next.end());
            
            double delta = target_val - q_current[exp.action];
            
            double h1[HIDDEN_SIZE] = {0};
            for(int j=0; j<HIDDEN_SIZE; ++j) 
                for(int i=0; i<INPUT_SIZE; ++i)
                    h1[j] += exp.state[i] * W1[i][j];
            
            double h2[HIDDEN_SIZE] = {0};
            for(int j=0; j<HIDDEN_SIZE; ++j)
                for(int i=0; i<HIDDEN_SIZE; ++i)
                    h2[j] += h1[i] * W2[i][j];
            
            for(int j=0; j<OUTPUT_SIZE; ++j) {
                double grad = (j == exp.action) ? delta : 0;
                for(int i=0; i<HIDDEN_SIZE; ++i) {
                    dW3[i][j] += grad * h2[i];
                    db3[j] += grad;
                }
            }
            
            for(int i=0; i<HIDDEN_SIZE; ++i) {
                double grad_h2 = 0;
                for(int j=0; j<OUTPUT_SIZE; ++j)
                    grad_h2 += delta * W3[i][j];
                grad_h2 *= (h2[i] > 0);
                
                for(int k=0; k<HIDDEN_SIZE; ++k) {
                    dW2[k][i] += grad_h2 * h1[k];
                    db2[i] += grad_h2;
                }
            }
            
            for(int i=0; i<HIDDEN_SIZE; ++i) {
                double grad_h1 = 0;
                for(int j=0; j<HIDDEN_SIZE; ++j)
                    grad_h1 += dW2[i][j] * (h1[j] > 0);
                
                for(int k=0; k<INPUT_SIZE; ++k) {
                    dW1[k][i] += grad_h1 * exp.state[k];
                    db1[i] += grad_h1;
                }
            }
        }
        
        for(int i=0; i<INPUT_SIZE; ++i)
            for(int j=0; j<HIDDEN_SIZE; ++j)
                W1[i][j] += LEARNING_RATE * dW1[i][j] / BATCH_SIZE;
        
        for(int i=0; i<HIDDEN_SIZE; ++i) {
            b1[i] += LEARNING_RATE * db1[i] / BATCH_SIZE;
            for(int j=0; j<HIDDEN_SIZE; ++j)
                W2[i][j] += LEARNING_RATE * dW2[i][j] / BATCH_SIZE;
            b2[i] += LEARNING_RATE * db2[i] / BATCH_SIZE;
        }
        
        for(int i=0; i<HIDDEN_SIZE; ++i)
            for(int j=0; j<OUTPUT_SIZE; ++j)
                W3[i][j] += LEARNING_RATE * dW3[i][j] / BATCH_SIZE;
        
        for(int j=0; j<OUTPUT_SIZE; ++j)
            b3[j] += LEARNING_RATE * db3[j] / BATCH_SIZE;
    }

    void save(const std::string& path) {
        std::ofstream file(path, std::ios::binary);
        file.write((char*)W1, sizeof(W1));
        file.write((char*)b1, sizeof(b1));
        file.write((char*)W2, sizeof(W2));
        file.write((char*)b2, sizeof(b2));
        file.write((char*)W3, sizeof(W3));
        file.write((char*)b3, sizeof(b3));
    }

    void load(const std::string& path) {
        std::ifstream file(path, std::ios::binary);
        file.read((char*)W1, sizeof(W1));
        file.read((char*)b1, sizeof(b1));
        file.read((char*)W2, sizeof(W2));
        file.read((char*)b2, sizeof(b2));
        file.read((char*)W3, sizeof(W3));
        file.read((char*)b3, sizeof(b3));
    }
};

std::vector<Experience> memory;
NeuralNetwork policy_net, target_net;
std::mt19937 gen(std::random_device{}());
double epsilon = EPS_START;

std::bitset<INPUT_SIZE> board_to_state(const std::vector<std::vector<char>>& board) {
    std::bitset<INPUT_SIZE> state;
    int idx = 0;
    for(int i=0; i<ROWS; ++i)
        for(int j=0; j<COLS; ++j) {
            state[idx++] = (board[i][j] == 'R');
            state[idx++] = (board[i][j] == 'Y');
        }
    return state;
}

int select_action(const std::bitset<INPUT_SIZE>& state, const NeuralNetwork& model, bool training) {
    std::vector<int> valid_actions;
    for(int col=0; col<COLS; ++col)
        if(state[col*2] == 0 && state[col*2+1] == 0)
            valid_actions.push_back(col);
    
    if(valid_actions.empty()) return -1;
    
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    if(training && dist(gen) < epsilon) {
        std::uniform_int_distribution<int> action_dist(0, valid_actions.size()-1);
        return valid_actions[action_dist(gen)];
    }
    
    auto q_values = model.forward(state);
    std::vector<std::pair<double, int>> action_values;
    for(int a : valid_actions)
        action_values.emplace_back(q_values[a], a);
    
    return std::max_element(action_values.begin(), action_values.end())->second;
}

bool check_win(const std::vector<std::vector<char>>& board, char player) {
    auto check_line = [&](int r, int c, int dr, int dc) {
        for(int i=0; i<4; ++i)
            if(board[r + i*dr][c + i*dc] != player) return false;
        return true;
    };
    
    for(int r=0; r<ROWS; ++r)
        for(int c=0; c<COLS-3; ++c)
            if(check_line(r, c, 0, 1)) return true;
    
    for(int c=0; c<COLS; ++c)
        for(int r=0; r<ROWS-3; ++r)
            if(check_line(r, c, 1, 0)) return true;
    
    for(int r=0; r<ROWS-3; ++r)
        for(int c=0; c<COLS-3; ++c)
            if(check_line(r, c, 1, 1)) return true;
    
    for(int r=3; r<ROWS; ++r)
        for(int c=0; c<COLS-3; ++c)
            if(check_line(r, c, -1, 1)) return true;
    
    return false;
}

void train_ai(int games) {
    _mkdir("memory");
    target_net = policy_net;
    int update_counter = 0;
    int wins = 0, losses = 0, draws = 0;
    auto training_start = std::chrono::steady_clock::now();
    
    for(int game=0; game<games; ++game) {
        std::vector<std::vector<char>> board(ROWS, std::vector<char>(COLS, ' '));
        char current = 'Y';
        std::vector<Experience> episode;
        bool done = false;
        char winner = ' ';
        
        while(!done) {
            auto state = board_to_state(board);
            int action = select_action(state, policy_net, true);
            
            if(action == -1) break;
            int row = ROWS-1;
            while(row >=0 && board[row][action] != ' ') row--;
            board[row][action] = current;
            
            double reward = 0;
            if(check_win(board, current)) {
                reward = (current == 'Y') ? 1.0 : -1.0;
                done = true;
                winner = current;
            }
            else if(std::all_of(board[0].begin(), board[0].end(), 
                [](char c) { return c != ' '; })) {
                done = true;
                draws++;
            }
            
            auto next_state = board_to_state(board);
            episode.push_back({state, action, reward, next_state, done});
            current = (current == 'Y') ? 'R' : 'Y';
        }
        
        if(winner == 'Y') wins++;
        else if(winner == 'R') losses++;
        
        for(auto& exp : episode) {
            if(memory.size() >= MEMORY_CAPACITY)
                memory[gen() % MEMORY_CAPACITY] = exp;
            else
                memory.push_back(exp);
        }
        
        if(memory.size() >= BATCH_SIZE) {
            std::vector<Experience> batch;
            std::sample(memory.begin(), memory.end(), std::back_inserter(batch),
                       BATCH_SIZE, gen);
            policy_net.update(target_net, batch);
            policy_net.soft_update(target_net);
        }
        
        epsilon = std::max(EPS_END, epsilon * EPS_DECAY);
        
        if((game+1) % 100 == 0) {
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - training_start).count();
            double games_per_sec = 100.0 / (elapsed > 0 ? elapsed : 1);
            
            std::cout << "Game " << std::setw(6) << game+1 << "/" << games 
                      << " | EPS: " << std::fixed << std::setprecision(4) << epsilon
                      << " | W/L/D: " << wins << "/" << losses << "/" << draws
                      << " | GPS: " << std::setprecision(1) << games_per_sec
                      << " | Mem: " << memory.size() << "     \r";
            std::cout.flush();
            
            wins = losses = draws = 0;
            training_start = now;
        }
        
        if((game+1) % 1000 == 0) {
            policy_net.save("memory/policy.bin");
            target_net.save("memory/target.bin");
        }
    }
    std::cout << "\nTraining complete. Models saved to memory/\n";
}

int ai_move(const std::vector<std::vector<char>>& board) {
    auto state = board_to_state(board);
    return select_action(state, policy_net, false);
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
        }
        else {
            int move = ai_move(board);
            int row = ROWS-1;
            while(row >=0 && board[row][move] != ' ') row--;
            board[row][move] = ai;
        }
        
        if(check_win(board, current)) {
            system("cls");
            display_board(board);
            std::cout << (current == human ? "Human" : "AI") << " wins!\n";
            break;
        }
        
        if(std::all_of(board[0].begin(), board[0].end(), 
            [](char c) { return c != ' '; })) {
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