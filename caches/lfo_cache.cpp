//
// Created by Arnav Garg on 2019-11-28.
//

#include <fstream>
#include "lfo_cache.h"

using namespace std;

double LFOCache::run_lightgbm(std::vector<double> feature) {

    if (boosterHandle == nullptr) {
        cout << "FUCK THIS UP." << endl;
    }

    static int count = 0;

    if (count % 1000001 == 0) {
        cout << "[+] Running the LightGBM model (Will only display once!)" << endl;
    }

    double* featureVector = new double[feature.size()];
    for (int i = 0; i < feature.size(); i++) {
        featureVector[i] = feature[i];
    }
    int64_t predictionsLength = 1;
    double predictions = 1;

    LGBM_BoosterPredictForMatSingleRow(boosterHandle,
                                       featureVector,
                                       C_API_DTYPE_FLOAT64,
                                       feature.size(),
                                       1,
                                       C_API_PREDICT_RAW_SCORE,
                                       -1,
                                       "",
                                       &predictionsLength,
                                       &predictions);

    if (predictionsLength != 1) {
        std::cout << "predictionsLength returned more than 1 value for input";
    }

    if (count % 1000001 == 0) {
        cout << "[+] LightGBM prediction complete (Will only display once!)" << endl;
    }
    count += 1;
    return predictions;
}

void LFOCache::train_lightgbm(std::vector<std::vector<double>> & features, std::vector<double> & opt_decisions) {

    int freedBooster = LGBM_BoosterFree(boosterHandle);

    if (freedBooster == 0) {
        cout << "[+] Freed Booster successfully" << std::endl;
    } else {
        cout << "[-] Freed Booster failed" << std::endl;
    }

    int numSamples = features.size();
    int featureLength = features[0].size();

    const char* training_data_filename = "../training_data_file.data";


    ofstream file;
    file.open(training_data_filename);

    file << fixed;

    cout << "[+] Starting training of LightGBM" << std::endl;
    for (int i = 0; i < numSamples; i++) {
        file << opt_decisions[i] << " ";
        for (int j = 0; j < features[0].size(); j++) {
            file << j << ":" << features[i][j];
            if (j != features[i].size()-1) {
                file << " ";
            }
        }
        file << endl;
    }

    file.close();

    int isdataSetLoaded = 0;

    if (dataHandle == nullptr){
        cout << "[+] Creating a dataset for LightGBM" << endl;
        isdataSetLoaded = LGBM_DatasetCreateFromFile(training_data_filename, "", nullptr, &dataHandle);
        cout << "[+] Created a dataset for LightGBM" << endl;
    }
    if (isdataSetLoaded != 0) {
        std::cout << "Loading dataset failed\n";
    }

    const char* parameters = "num_iterations=100 num_threads=4";

    int isLearnerCreated = LGBM_BoosterCreate(dataHandle, parameters, &boosterHandle);

    if (isLearnerCreated != 0) {
        std::cout << "Creating learner failed\n";
    }

    for (int i=0 ; i < numIterations; i++) {
        int isFinished;
        int isUpdated = LGBM_BoosterUpdateOneIter(boosterHandle, &isFinished);
        if (isUpdated != 0) {
            std::cout << "Failed to update at iteration number " << i << "\n";
        }
        if (isFinished == 1) {
            std::cout << "No further gain, cannot split anymore" << std::endl;
            break;
        }
    }
    dataHandle = nullptr;
    LGBM_BoosterSaveModel(boosterHandle, 0, -1, "../booster.data");

    int freedData = LGBM_DatasetFree(dataHandle);

    if (freedData == 0) {
        cout << "[+] Free DataHandle successfully" << std::endl;
    } else {
        cout << "[-] Free DataHandle failed" << std::endl;
    }
    cout << "[+] Training of LightGBM completed" << std::endl;
}


bool LFOCache::lookup(SimpleRequest* req) {
    auto it = _cacheMap.find(req->getId());
    if(it != _cacheMap.end()) return true;
    return false;
};

void LFOCache::admit(SimpleRequest* req) {
    const uint64_t size = req->getSize();
    auto lfoFeature = get_lfo_feature(req);
    double dvar = run_lightgbm(lfoFeature.get_vector());
    if (dvar >= threshold) {
        while (_currentSize + size > _cacheSize) {
            evict();
        }

        CacheObject obj(req);
        obj.dvar = dvar;
        _cacheMap.insert({lfoFeature.id, obj});
        _cacheObjectMinpq.push(obj);
        _currentSize += size;
    }

};

void LFOCache::evict(SimpleRequest* req) {
    throw "Random eviction not supported for LFOCache";
};

void LFOCache::evict() {
    evict_return();
}

SimpleRequest* LFOCache::evict_return() {
    if(_cacheObjectMinpq.size() > 0){
        CacheObject obj = _cacheObjectMinpq.top();
        _currentSize -= obj.size;
        _cacheMap.erase(obj.id);
        _cacheObjectMinpq.pop();
        SimpleRequest *req = new SimpleRequest(obj.id,obj.size);
        return req;
    }
    return NULL;
}
