#include "ThreadPool.h"

ThreadPool::ThreadPool(uint32_t workerCount)
{
    start(workerCount);
}

ThreadPool::~ThreadPool()
{
    stop();
}

void ThreadPool::start(uint32_t workerCount)
{
    stop();

    if (workerCount == 0)
    {
        workerCount = 1;
    }

    stopping_.store(false, std::memory_order_release);
    workers_.reserve(workerCount);

    for (uint32_t i = 0; i < workerCount; ++i)
    {
        workers_.emplace_back([this]()
        {
            while (true)
            {
                std::function<void()> task;
                {
                    std::unique_lock lock(mutex_);
                    cv_.wait(lock, [this]()
                    {
                        return stopping_.load(std::memory_order_acquire) || !tasks_.empty();
                    });

                    if (stopping_.load(std::memory_order_acquire) && tasks_.empty())
                    {
                        return;
                    }

                    task = std::move(tasks_.front());
                    tasks_.pop();
                }

                task();
            }
        });
    }
}

void ThreadPool::stop()
{
    {
        std::scoped_lock lock(mutex_);
        stopping_.store(true, std::memory_order_release);
    }
    cv_.notify_all();

    for (auto& worker : workers_)
    {
        if (worker.joinable())
        {
            worker.join();
        }
    }
    workers_.clear();

    {
        std::scoped_lock lock(mutex_);
        std::queue<std::function<void()>> empty;
        tasks_.swap(empty);
    }
}

uint32_t ThreadPool::workerCount() const
{
    return static_cast<uint32_t>(workers_.size());
}
