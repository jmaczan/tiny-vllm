#include "prefix_cache.h"

#include <cassert>
#include <iostream>

PrefixCacheManager::PrefixCacheManager(size_t max_blocks)
    : max_blocks_(max_blocks) {}

uint64_t PrefixCacheManager::hashPrefix(const std::vector<int>& tokens,
                                        size_t prefix_len,
                                        int layer) const {
    uint64_t h = 14695981039346656037ull;  // FNV-1a offset basis
    size_t limit = std::min(prefix_len, tokens.size());

    h ^= static_cast<uint64_t>(layer + 1);
    h *= 1099511628211ull;

    for (size_t i = 0; i < limit; ++i) {
        h ^= static_cast<uint64_t>(tokens[i] + 1);
        h *= 1099511628211ull;
    }
    return h;
}

int PrefixCacheManager::allocateBlock() {
    if (blocks_.size() >= max_blocks_) {
        return -1;
    }
    blocks_.push_back(PrefixCacheEntry{});
    return static_cast<int>(blocks_.size() - 1);
}

int PrefixCacheManager::lookupOrInsertPrefix(const std::vector<int>& tokens,
                                             size_t prefix_len,
                                             int layer,
                                             int block_id) {
    uint64_t key = hashPrefix(tokens, prefix_len, layer);
    auto it = prefix_map_.find(key);
    if (it != prefix_map_.end()) {
        return it->second;
    }

    int entry = allocateBlock();
    if (entry < 0) {
        return -1;
    }

    int stored_block_id = block_id >= 0 ? block_id : entry;
    blocks_[entry].hash = key;
    blocks_[entry].block_id = stored_block_id;
    blocks_[entry].token_count = std::min(prefix_len, tokens.size());

    prefix_map_[key] = stored_block_id;
    return stored_block_id;
}

int PrefixCacheManager::lookupPrefix(const std::vector<int>& tokens,
                                     size_t prefix_len,
                                     int layer) const {
    uint64_t key = hashPrefix(tokens, prefix_len, layer);
    auto it = prefix_map_.find(key);
    if (it != prefix_map_.end()) {
        return it->second;
    }
    return -1;
}

void PrefixCacheManager::releaseBlock(int block_id) {
    if (block_id < 0 || block_id >= static_cast<int>(blocks_.size())) {
        return;
    }
    blocks_[block_id].hash = 0;
    blocks_[block_id].block_id = -1;
    blocks_[block_id].token_count = 0;
}