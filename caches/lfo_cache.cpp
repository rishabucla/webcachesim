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

    return predictions;
}

void LFOCache::train_lightgbm(std::vector<std::vector<double>> & features, std::vector<double> & opt_decisions) {

    int freedBooster = LGBM_BoosterFree(boosterHandle);
    static int counter = 0;

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

    const char* parameters = "num_iterations=30 num_threads=4";

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
    string filepath = "../booster_data/booster_" + to_string(counter++) + ".data";
    LGBM_BoosterSaveModel(boosterHandle, 0, -1, filepath.data());

    int freedData = LGBM_DatasetFree(dataHandle);

    if (freedData == 0) {
        cout << "[+] Free DataHandle successfully" << std::endl;
    } else {
        cout << "[-] Free DataHandle failed" << std::endl;
    }
    cout << "[+] Training of LightGBM completed" << std::endl;
}

double LFOCache::run_rvm(std::vector<double> feature) {
    sample_type sample;

    for(auto j=0; j<feature.size(); j++){
        sample(j) = feature.at(j);
    }

    return rvm_learned_function(sample);
}

/**
 * Reference: http://dlib.net/rvm_ex.cpp.html
 * @param features
 * @param labels
 */
void LFOCache::train_rvm(std::vector<std::vector<double>> features, std::vector<double> labels) {
    std::vector<sample_type> samples;

    for(auto i = 0; i < features.size(); i++){
        sample_type samp;
        samp.set_size(features.size());

        auto feature = features.at(i);

        for(auto j=0; j<feature.size(); j++){
            samp(j) = feature.at(j);
        }

        samples.push_back(samp);
    }

    dlib::vector_normalizer<sample_type> normalizer;
    // let the normalizer learn the mean and standard deviation of the samples
    normalizer.train(samples);
    // now normalize each sample
    for (unsigned long i = 0; i < samples.size(); ++i)
        samples[i] = normalizer(samples[i]);

    dlib::krr_trainer<kernel_type> trainer;

    if(rvm_cross_validate){
        cout << "doing cross validation" << endl;
        double max = 0;
        for (double gamma = 0.000001; gamma <= 1; gamma *= 5)
        {
            // tell the trainer the parameters we want to use
            trainer.set_kernel(kernel_type(gamma));

            // Print out the cross validation accuracy for 3-fold cross validation using the current gamma.
            // cross_validate_trainer() returns a row vector.  The first element of the vector is the fraction
            // of +1 training examples correctly classified and the second number is the fraction of -1 training
            // examples correctly classified.

            auto cross_validation_results = cross_validate_trainer(trainer, samples, labels, 3);

            double class1 = *(cross_validation_results.begin());
            double class2 = *(cross_validation_results.begin() + 1);

            if(class1 + class2 > max){
                max = class1 + class2;
                rvm_gamma = gamma;
            }
        }

        cout << "Best RVM gamma: " << rvm_gamma << endl;
        rvm_cross_validate = false;
    }

    trainer.set_kernel(kernel_type(rvm_gamma));

    // Here we are making an instance of the normalized_function object.  This object provides a convenient
    // way to store the vector normalization information along with the decision function we are
    // going to learn.
    rvm_learned_function.normalizer = normalizer;  // save normalization information
    rvm_learned_function.function = trainer.train(samples, labels); // perform the actual RVM training and save the results
}

double LFOCache::run_svm(std::vector<double> feature) {
    sample_type sample;

    for(auto j=0; j<feature.size(); j++){
        sample(j) = feature.at(j);
    }

    return svm_learned_function(sample);
}

/**
 * Reference: http://dlib.net/svm_c_ex.cpp.html
 * @param features
 * @param labels
 */
void LFOCache::train_svm(std::vector<std::vector<double>> features, std::vector<double> labels) {
    std::vector<sample_type> samples;

    for(auto i = 0; i < features.size(); i++){
        sample_type samp;
        samp.set_size(features.size());

        auto feature = features.at(i);

        for(auto j=0; j<feature.size(); j++){
            samp(j) = feature.at(j);
        }

        samples.push_back(samp);
    }

    dlib::vector_normalizer<sample_type> normalizer;
    // let the normalizer learn the mean and standard deviation of the samples
    normalizer.train(samples);
    // now normalize each sample
    for (unsigned long i = 0; i < samples.size(); ++i)
        samples[i] = normalizer(samples[i]);

    randomize_samples(samples, labels);

    // here we make an instance of the svm_c_trainer object that uses our kernel
    // type.
    dlib::svm_c_trainer<kernel_type> trainer;

    if(svm_cross_validate){
        double max = 0;
        for (double gamma = 0.00001; gamma <= 1; gamma *= 5)
        {
            for (double C = 1; C < 100000; C *= 5)
            {
                // tell the trainer the parameters we want to use
                trainer.set_kernel(kernel_type(gamma));
                trainer.set_c(C);

                cout << "gamma: " << gamma << "    C: " << C;
                // Print out the cross validation accuracy for 3-fold cross validation using
                // the current gamma and C.  cross_validate_trainer() returns a row vector.
                // The first element of the vector is the fraction of +1 training examples
                // correctly classified and the second number is the fraction of -1 training
                // examples correctly classified.
                auto cross_validation_results = cross_validate_trainer(trainer, samples, labels, 3);

                double class1 = *(cross_validation_results.begin());
                double class2 = *(cross_validation_results.begin() + 1);

                if(class1 + class2 > max){
                    max = class1 + class2;
                    svm_gamma = gamma;
                    svm_c = C;
                }
            }
        }

        cout << "Best SVM gamma: " << svm_gamma << " C: " <<  svm_c << endl;
        svm_cross_validate = false;
    }

    trainer.set_kernel(kernel_type(svm_gamma));
    trainer.set_c(svm_c);
    typedef dlib::decision_function<kernel_type> dec_funct_type;
    typedef dlib::normalized_function<dec_funct_type> funct_type;

    // Here we are making an instance of the normalized_function object.  This
    // object provides a convenient way to store the vector normalization
    // information along with the decision function we are going to learn.
    svm_learned_function.normalizer = normalizer;  // save normalization information
    svm_learned_function.function = trainer.train(samples, labels); // perform the actual SVM training and save the results

}


bool LFOCache::lookup(SimpleRequest* req) {
    auto it = _cacheMap.find(req->getId());
    if(it != _cacheMap.end()) return true;
    return false;
};

void LFOCache::admit(SimpleRequest* req) {
    const uint64_t size = req->getSize();
    double dvar = run_lightgbm(req->getFeatureVector());
    if (dvar >= threshold) {
        while (_currentSize + size > _cacheSize) {
            evict();
        }

        CacheObject obj(req);
        obj.dvar = dvar;
        _cacheMap.insert({req->getId(), obj});
        _cacheObjectMinpq.push(obj);
        _currentSize += size;
    }

};

void LFOCache::evict(SimpleRequest* req) {
    throw "Random eviction not supported for LFOCache";
};

void LFOCache::evict() {
//    evict_return();
    if(_cacheObjectMinpq.size() > 0){
        CacheObject obj = _cacheObjectMinpq.top();
        _currentSize -= obj.size;
        _cacheMap.erase(obj.id);
        _cacheObjectMinpq.pop();
    }
}

SimpleRequest* LFOCache::evict_return() {
    return NULL;
}
