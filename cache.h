#ifndef CACHE_H
#define CACHE_H

#include <unordered_set>
#include <unordered_map>
#include <map>
#include <iostream>
#include <string>
#include <vector>
#include <cstdint>
#include <memory>
#include "request.h"
#include "lfo_features.h"

// uncomment to enable cache debugging:
// #define CDEBUG 1

// util for debug
#ifdef CDEBUG
inline void logMessage(std::string m, double x, double y, double z) {
    std::cerr << m << "," << x << "," << y  << "," << z << "\n";
}
#define LOG(m,x,y,z) logMessage(m,x,y,z)
#else
#define LOG(m,x,y,z)
#endif



class Cache;

class CacheFactory {
public:
    CacheFactory() {}
    virtual std::unique_ptr<Cache> create_unique() = 0;
};

class Cache {
    std::unordered_map<IdType, LFOFeature> id2feature;
    std::unordered_map<IdType, LFOFeature> id2RlFeature;
    bool useRLCacheFeatures;
    bool useExponentialTimeGap;

    uint64_t requestsSoFar;

    LFOFeature getRLFeature(SimpleRequest* r){
        auto it = id2RlFeature.find(r->getId());

        requestsSoFar+=1;

        if(it != id2RlFeature.end()){
            LFOFeature prevLFOFeature = it->second;

            LFOFeature newLfoFeature(r->getId(), r->getSize(), r->getTimeStamp());
            update_timegaps(newLfoFeature, r->getTimeStamp());
            newLfoFeature.available_cache_size = getFreeBytes();
            newLfoFeature.request_no = requestsSoFar;
            newLfoFeature.ordinal_recency =  requestsSoFar - prevLFOFeature.request_no + 1;
            newLfoFeature.temporal_recency = r->getTimeStamp() - prevLFOFeature.timestamp;
            newLfoFeature.times_requested = 1+prevLFOFeature.times_requested;
            newLfoFeature.frequency = newLfoFeature.times_requested / requestsSoFar;
            newLfoFeature.calculateDeltaJ();
            newLfoFeature.calculateRhoJ();
            newLfoFeature.use_exponential_time_gap = useExponentialTimeGap;
            newLfoFeature.use_rl_cache_features = useRLCacheFeatures;

            id2RlFeature.insert({r->getId(), newLfoFeature});

        }else{
            std::vector<uint64_t> newTimeGapList;
            LFOFeature lfoFeature(r->getId(), r->getSize(), r->getTimeStamp());
            lfoFeature.request_no = requestsSoFar;
            lfoFeature.available_cache_size = getFreeBytes();
            lfoFeature.ordinal_recency = 0;//new request, ordinal recency is 0
            lfoFeature.temporal_recency = 0;//new request, temporal recency is 0
            lfoFeature.times_requested = 1;
            lfoFeature.frequency = (1.0/requestsSoFar); //new request, fraction of requests so far
            lfoFeature.calculateRhoJ();
            lfoFeature.calculateDeltaJ();
            lfoFeature.use_exponential_time_gap = useExponentialTimeGap;
            lfoFeature.use_rl_cache_features = useRLCacheFeatures;

            id2RlFeature.insert({r->getId(), lfoFeature});

        }
        return id2RlFeature[r->getId()];

    }
public:
    // create and destroy a cache
    Cache()
        : _cacheSize(0),
          _currentSize(0)
    {
        requestsSoFar = 0;
        useRLCacheFeatures = false;
        useExponentialTimeGap = false;
    }
    virtual ~Cache(){};

    // main cache management functions (to be defined by a policy)
    virtual bool lookup(SimpleRequest* req) = 0;
    virtual void admit(SimpleRequest* req) = 0;
    virtual void evict(SimpleRequest* req) = 0;
    virtual void evict() = 0;

    // configure cache parameters
    virtual void setSize(uint64_t cs) {
        _cacheSize = cs;
        while (_currentSize > _cacheSize) {
            evict();
        }
    }
    virtual void setPar(std::string parName, std::string parValue) {}

    uint64_t getCurrentSize() const {
        return(_currentSize);
    }
    uint64_t getSize() const {
        return(_cacheSize);
    }

    uint64_t getFreeBytes() const {
        return (_cacheSize - _currentSize);
    }

    void reset(){
        requestsSoFar = 0;
    }

    // helper functions (factory pattern)
    static void registerType(std::string name, CacheFactory *factory) {
        get_factory_instance()[name] = factory;
    }
    static std::unique_ptr<Cache> create_unique(std::string name) {
        std::unique_ptr<Cache> Cache_instance;
        if(get_factory_instance().count(name) != 1) {
            std::cerr << "unkown cacheType" << std::endl;
            return nullptr;
        }
        Cache_instance = get_factory_instance()[name]->create_unique();
        return Cache_instance;
    }

    // Here is where I am keeping all the functions that are needed by the
    // LFO Cache. I don't really care right now. Maybe think of a better way to handle this later.
    // ___________________________

    void update_timegaps(LFOFeature & feature, uint64_t new_time) {
        uint64_t time_diff = new_time - feature.timestamp;

        for (auto it = feature.timegaps.begin(); it != feature.timegaps.end(); it++) {
            *it = *it + time_diff;
        }

        feature.timegaps.push_back(time_diff);

        if(feature.use_exponential_time_gap){
            if(feature.timegaps.size() > 2050){
                feature.timegaps.erase(feature.timegaps.begin());
            }
        }else{
            if(feature.timegaps.size() > 50){
                feature.timegaps.erase(feature.timegaps.begin());
            }
        }
    }

    LFOFeature get_lfo_feature(SimpleRequest* req) {
        if(useRLCacheFeatures) return getRLFeature(req);

        if (id2feature.find(req->getId()) != id2feature.end()) {
            LFOFeature& feature = id2feature[req->getId()];
            update_timegaps(feature, req->getTimeStamp());
            feature.timestamp = req->getTimeStamp();
            feature.available_cache_size = getFreeBytes();
        } else {
            LFOFeature feature(req->getId(), req->getSize(), req->getTimeStamp());
            feature.use_exponential_time_gap = useExponentialTimeGap;
            feature.use_rl_cache_features = useRLCacheFeatures;
            feature.available_cache_size = getFreeBytes();
            id2feature[req->getId()] = feature;
        }

        return id2feature[req->getId()];
    }

    virtual void train_lightgbm(std::vector<std::vector<double>> & features, std::vector<double> & labels) {}
//    virtual double run_lightgbm(std::vector<double> feature);
    // _____________________________

protected:
    // basic cache properties
    uint64_t _cacheSize; // size of cache in bytes
    uint64_t _currentSize; // total size of objects in cache in bytes

    // helper functions (factory pattern)
    static std::map<std::string, CacheFactory *> &get_factory_instance() {
        static std::map<std::string, CacheFactory *> map_instance;
        return map_instance;
    }
};

template<class T>
class Factory : public CacheFactory {
public:
    Factory(std::string name) { Cache::registerType(name, this); }
    std::unique_ptr<Cache> create_unique() {
        std::unique_ptr<Cache> newT(new T);
        return newT;
    }
};


#endif /* CACHE_H */
