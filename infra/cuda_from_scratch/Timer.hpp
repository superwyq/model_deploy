 #include <iostream>
 #include <chrono>

 class Timer {
    public:
        Timer() = default;
        ~Timer() = default;

        void start(){
            start_time = std::chrono::high_resolution_clock::now();
        }

        void end(){
            end_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<float> duration = end_time - start_time;
            std::cout << "Time: " << duration.count() * 1000 << "ms" << std::endl;
        }

    private:
        std::chrono::time_point<std::chrono::high_resolution_clock> start_time, end_time;
 };