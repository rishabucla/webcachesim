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
               unique_ptr<Cache> & webcache,
               ifstream & infile,
               size_t batch_size,
               ofstream & outfile) {

    uint64_t time, id, size;
    uint64_t counter = 0;
    uint64_t hit = 0;

    while (!infile.eof()) {

        if (counter >= batch_size) {
            break;
        }

        infile >> time >> id >> size;

        SimpleRequest req(id, size, time);
        prev_requests[counter] = req;

        vector<double> prev_feature = webcache->get_lfo_feature(&req).get_vector();
        if (!prev_feature.empty()) {
            prev_features[counter] = prev_feature;
        }

        if (webcache->lookup(&req)) {
            hit++;
            outfile << id << ' ' << size << ' ' << '1' << ' ' << '1' << std::endl;
        } else {
            webcache->admit(&req);
            if (webcache->lookup(&req)) {
                outfile << id << ' ' << size << ' ' << '1' << ' ' << '0' << std::endl;
            } else {
                outfile << id << ' ' << size << ' ' << '0' << ' ' << '0' << std::endl;
            }
        }
        counter += 1;
    }

    return hit;

}

void run_simulation(const string path, const string cacheType, const uint64_t cache_size) {
    unique_ptr<Cache> webcache = Cache::create_unique(cacheType);
    if(webcache == nullptr)
        exit(0);

    // configure cache size
    webcache->setSize(cache_size);

    ifstream infile;
    std::ofstream outfile;
    size_t batch_size = 100000;
    bool changed_to_lfo = false;

    vector<SimpleRequest> prev_requests(batch_size);
    vector<vector<double >> prev_features(batch_size);

    size_t iterations = 0;

    infile.open(path);
    outfile.open("../cache_decisions.out");
    while (!infile.eof()) {


        cout << "Iteration: " << iterations << std::endl;
        uint64_t hits = run_model(prev_requests, prev_features, webcache, infile, batch_size, outfile);
        cout << "[+] Number of hits: " << hits << "\n";

        if (prev_features.size() != 0 && prev_requests.size() != 0 && prev_features.size() == prev_requests.size()){
            cout << "[+] Computing optimal decisions"<< std::endl;
            vector<double> optimal_decisions = getOptimalDecisions(prev_requests, webcache->getSize());

            if (!changed_to_lfo) {
                webcache = Cache::create_unique("LFO");
                webcache->setSize(cache_size);
                changed_to_lfo = !changed_to_lfo;
            }

            webcache->train_lightgbm(prev_features, optimal_decisions);
//            prev_features.clear();
//            prev_requests.clear();
        }

        iterations += 1;
    }
}

int main (int argc, char* argv[])
{

    // output help if insufficient params
    if(argc < 4) {
        cerr << "webcachesim traceFile cacheType cacheSizeBytes" << endl;
        return 1;
    }

    const char* path = argv[1];
    const string cacheType = argv[2];
    const uint64_t cache_size  = std::stoull(argv[3]);



    run_simulation(path, cacheType, cache_size);


    return 0;
}
