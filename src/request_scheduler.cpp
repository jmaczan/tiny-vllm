#include "request_scheduler.h"

#include <algorithm>
#include <iostream>

RequestScheduler::RequestScheduler(PrefixCacheManager* cache, size_t chunk_size)
    : cache_(cache), chunk_size_(chunk_size) {}

void RequestScheduler::addRequest(const Request& request) {
    waiting_queue_.push_back(request);
}

Request* RequestScheduler::startNextWaitingRequest(int slot) {
    if (waiting_queue_.empty()) {
        return nullptr;
    }

    Request req = waiting_queue_.front();
    waiting_queue_.erase(waiting_queue_.begin());
    req.slot = slot;
    req.state = RequestState::Prefill;
    running_queue_.push_back(req);
    return &running_queue_.back();
}

Request* RequestScheduler::getRequestById(int request_id) {
    for (auto &req : running_queue_) {
        if (req.request_id == request_id) {
            return &req;
        }
    }
    for (auto &req : waiting_queue_) {
        if (req.request_id == request_id) {
            return &req;
        }
    }
    for (auto &req : finished_queue_) {
        if (req.request_id == request_id) {
            return &req;
        }
    }
    return nullptr;
}

bool RequestScheduler::finalizeRunningRequest(int request_id) {
    for (auto it = running_queue_.begin(); it != running_queue_.end(); ++it) {
        if (it->request_id == request_id) {
            it->state = RequestState::Finished;
            finished_queue_.push_back(*it);
            running_queue_.erase(it);
            return true;
        }
    }
    return false;
}

std::vector<StepResult> RequestScheduler::step() {
    std::vector<StepResult> results;

    // 1) 先处理 waiting -> prefill
    for (auto it = waiting_queue_.begin(); it != waiting_queue_.end();) {
        Request& req = *it;

        if (req.prefill_cursor >= req.tokens.size()) {
            req.state = RequestState::Decoding;
            running_queue_.push_back(req);
            it = waiting_queue_.erase(it);
            continue;
        }

        size_t remain = req.tokens.size() - req.prefill_cursor;
        size_t chunk_len = std::min(chunk_size_, remain);

        size_t prefix_len = req.prefill_cursor + chunk_len;
        int block_id = cache_->lookupOrInsertPrefix(req.tokens, prefix_len, 0);
        if (block_id >= 0) {
            req.block_ids.push_back(block_id);
        }

        req.prefill_cursor += chunk_len;
        req.state = (req.prefill_cursor >= req.tokens.size())
                        ? RequestState::Decoding
                        : RequestState::Prefill;

        results.push_back(StepResult{
            req.request_id,
            block_id,
            req.prefill_cursor,
            req.decode_len,
            req.state
        });

        if (req.state == RequestState::Decoding) {
            running_queue_.push_back(req);
            it = waiting_queue_.erase(it);
        } else {
            ++it;
        }
    }

    // 2) 再处理 decode phase
    for (auto it = running_queue_.begin(); it != running_queue_.end();) {
        Request& req = *it;
        req.decode_len += 1;

        // 这里可以替换成真实 kernel 调用，如 kernels.cu 中的 decode kernel
        // 这里只做最小模拟
        if (req.decode_len >= req.tokens.size()) {
            req.state = RequestState::Finished;
            finished_queue_.push_back(req);
            it = running_queue_.erase(it);
        } else {
            req.state = RequestState::Decoding;
            ++it;
        }

        results.push_back(StepResult{
            req.request_id,
            -1,
            req.prefill_cursor,
            req.decode_len,
            req.state
        });
    }

    return results;
}

size_t RequestScheduler::waitingCount() const {
    return waiting_queue_.size();
}

size_t RequestScheduler::runningCount() const {
    return running_queue_.size();
}

size_t RequestScheduler::finishedCount() const {
    return finished_queue_.size();
}