#include <fstream>
#include <string>
#include <regex>
#include "caches/lru_variants.h"
//#include "caches/gd_variants.h"
#include "caches/lfo_cache.h"
#include "request.h"
#include "caches/optimal.h"

using namespace std;

uint64_t run_model(vector<SimpleRequest> & prev_requests,
               vector<vector<double>> & prev_features,
               vector<dlib::matrix<double, 0, 1>> & prev_samples,
               unique_ptr<Cache> & webcache,
               ifstream & infile,
               size_t batch_size,
               ofstream & outfile) {

    uint64_t time, id, size;
    uint64_t counter = 0;
    uint64_t hit = 0;

    bool useFeatures = true;

    if(prev_samples.size() == batch_size) useFeatures = false;

    while (!infile.eof()) {

        if (counter >= batch_size) {
            break;
        }

        infile >> time >> id >> size;

        SimpleRequest req(id, size, time);
        prev_requests[counter] = req;

        if(useFeatures){
            vector<double> prev_feature = webcache->get_lfo_feature(&req).get_vector();
            if (!prev_feature.empty()) {
                prev_features[counter] = prev_feature;
            }

            if (webcache->lookup(&req)) {
                hit++;
                outfile << id << ' ' << size << ' ' << '1' << ' ' << '1' << std::endl;
            } else {
                req.setFeatureVector(prev_feature);
                webcache->admit(&req);
                if (webcache->lookup(&req)) {
                    outfile << id << ' ' << size << ' ' << '1' << ' ' << '0' << std::endl;
                } else {
                    outfile << id << ' ' << size << ' ' << '0' << ' ' << '0' << std::endl;
                }
            }
        }else{
            dlib::matrix<double, 0, 1> prev_sample = webcache->get_lfo_feature(&req).get_sample_type();
            prev_samples[counter] = prev_sample;

            if (webcache->lookup(&req)) {
                hit++;
                outfile << id << ' ' << size << ' ' << '1' << ' ' << '1' << std::endl;
            } else {
                req.setSampleType(prev_sample);
                webcache->admit(&req);
                if (webcache->lookup(&req)) {
                    outfile << id << ' ' << size << ' ' << '1' << ' ' << '0' << std::endl;
                } else {
                    outfile << id << ' ' << size << ' ' << '0' << ' ' << '0' << std::endl;
                }
            }
        }

        counter += 1;
    }

    return hit;

}

void run_simulation(const string path, const string cacheType, const uint64_t cache_size,
        bool use_exponential_time_gap, bool use_rl_cache, ml_model model) {
    unique_ptr<Cache> webcache = Cache::create_unique(cacheType);
    if(webcache == nullptr)
        exit(0);

    // configure cache size
    webcache->setSize(cache_size);
    webcache->setUseExponentialTimeGap(use_exponential_time_gap);
    webcache->setUseRLCacheFeatures(use_rl_cache);
    webcache->setMLModel(model);

    ifstream infile;

    std::ofstream outfile;

    size_t batch_size = 1000000;
    bool changed_to_lfo = false;

    bool read_file_optimal = true;
    string od_filepath = "../opt.decisions";
    ifstream opt_infile;

    vector<SimpleRequest> prev_requests(batch_size);
//    vector<vector<double >> prev_features(batch_size);

    vector<vector<double >> prev_features;//(batch_size);
    vector<dlib::matrix<double, 0, 1>> prev_samples;//(batch_size);

    if(model == SVM || model == RVM) prev_samples.resize(batch_size);
    if(model == LIGHT_GBM) prev_features.resize(batch_size);


    size_t iterations = 0;

    infile.open(path);
    outfile.open("../cache_decisions.out");
    opt_infile.open(od_filepath);
    while (!infile.eof()) {


        cout << "Iteration: " << iterations << std::endl;
        webcache->reset();
        uint64_t hits = run_model(prev_requests, prev_features, prev_samples, webcache, infile, batch_size, outfile);

        cout << "[+] Number of hits: " << hits << "\n";

        if (iterations == 49) {
            return;
        }

        if (
                (prev_features.size() != 0 && prev_requests.size() != 0 && prev_features.size() == prev_requests.size())
                ||
                (prev_samples.size() != 0 && prev_requests.size() != 0 && prev_samples.size() == prev_requests.size())
                ){


            if (!changed_to_lfo) {
                webcache = Cache::create_unique("LFO");
                webcache->setSize(cache_size);
                webcache->setUseExponentialTimeGap(use_exponential_time_gap);
                webcache->setUseRLCacheFeatures(use_rl_cache);
                webcache->setMLModel(model);
                changed_to_lfo = !changed_to_lfo;
            }

            if (iterations % 10 == 0) {
                cout << "[+] Computing optimal decisions"<< std::endl;
                vector<double> optimal_decisions;
                if (read_file_optimal) {
                    optimal_decisions = getOptimalDecisionsFromFile(batch_size, opt_infile);
                } else {
                    optimal_decisions = getOptimalDecisions(prev_requests, webcache->getSize());
                }
                cout << "[+] Calling Train Light GBM at iteration " << iterations << endl;
                webcache->train_model(prev_features, prev_samples, optimal_decisions);
            }
//            prev_features.clear();
//            prev_requests.clear();
        }

        webcache->clear_features();
        iterations += 1;
    }
}

void writeOptDecisions(const char* path, uint64_t batch_size, uint64_t cache_size, int metric){
    char fileName[100];
    sprintf (fileName, "../opt_decisions_%d_%llu_%llu", metric, batch_size, cache_size);
    std::ofstream outfile;
    outfile.open(fileName);

    std::ifstream infile;
    infile.open(path);

    uint64_t time, id, size;
    uint64_t counter = 0;
    vector<SimpleRequest> requests(batch_size);

    while (!infile.eof()) {

        while(counter <= batch_size && !infile.eof()){
            infile >> time >> id >> size;
            SimpleRequest req(id, size, time);
            requests[counter] = req;
            counter += 1;
        }

        //write optimal decisions
        vector<double> dvars = getOptimalDecisions(requests, cache_size);
        for (vector<double>::iterator it = dvars.begin() ; it != dvars.end(); ++it)
            outfile << *it << std::endl;

        counter = 0;
    }
}

ml_model getModel(int val){
    switch (val){
        case 1: return LIGHT_GBM;
        case 2: return RVM;
        case 3: return SVM;
    }
    //default
    return LIGHT_GBM;
}


int main (int argc, char* argv[])
{
    if(strcmp(argv[1], "optgen") == 0){
        //"bin optgen traceFile cacheSizeBytes batchSize metric"
        const char* path = argv[2];
        const uint64_t cache_size  = std::stoull(argv[3]);
        const uint64_t batch_size  = std::stoull(argv[4]);
        const int metric = std::stoull(argv[5]); //1 => BHR, 2 => OHR

        writeOptDecisions(path, batch_size, cache_size, metric);

    }else{
        // output help if insufficient params
        if(argc < 4) {
            cerr << "webcachesim traceFile cacheType cacheSizeBytes [use_exp_timegap] [use_rl_cache] [ml_model]" << endl;
            return 1;
        }

        const char* path = argv[1];
        const string cacheType = argv[2];
        const uint64_t cache_size  = std::stoull(argv[3]);

        bool use_exponential_time_gap = false, use_rl_cache = false;
        ml_model model = LIGHT_GBM;

        if(argc == 6){
            use_exponential_time_gap = std::stoull(argv[4]);
            use_rl_cache = std::stoull(argv[5]);
        }else if(argc == 7){
            use_exponential_time_gap = std::stoull(argv[4]);
            use_rl_cache = std::stoull(argv[5]);
            model = getModel(std::stoull(argv[6]));
        }

        run_simulation(path, cacheType, cache_size, use_exponential_time_gap, use_rl_cache, model);

        return 0;
    }

}

