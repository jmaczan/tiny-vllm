#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

struct PrefixCacheEntry {
    uint64_t hash = 0;
    int block_id = -1;
    size_t token_count = 0;
};

class PrefixCacheManager {
public:
    explicit PrefixCacheManager(size_t max_blocks = 1024);

    // 以 prefix_len 长度为单位，查找或创建 KV block
    // 如果 block_id >= 0，则在未命中时插入该物理块号
    int lookupOrInsertPrefix(const std::vector<int>& tokens, size_t prefix_len, int layer, int block_id = -1);

    // 仅查找，不创建
    int lookupPrefix(const std::vector<int>& tokens, size_t prefix_len, int layer) const;

    // 释放 block
    void releaseBlock(int block_id);

    size_t blockCount() const { return blocks_.size(); }

private:
    uint64_t hashPrefix(const std::vector<int>& tokens, size_t prefix_len, int layer) const;
    int allocateBlock();

private:
    size_t max_blocks_ = 1024;
    std::vector<PrefixCacheEntry> blocks_;
    std::unordered_map<uint64_t, int> prefix_map_;
};