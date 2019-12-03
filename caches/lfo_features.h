//
// Created by Arnav Garg on 2019-11-28.
//

#ifndef WEBCDN_LFO_FEATURES_H
#define WEBCDN_LFO_FEATURES_H

#include <request.h>
#include <vector>
#include "lib/alglib/src/stdafx.h"
#include "lib/alglib/src/dataanalysis.h"

enum OptimizationGoal {
    OBJECT_HIT_RATIO,
    BYTE_HIT_RATIO
};
const double ALPHA = 0.5;


struct LFOFeature {
    IdType id;
    uint64_t size;
    OptimizationGoal optimizationGoal;
    uint64_t timestamp;
    std::vector<uint64_t> timegaps;
    uint64_t available_cache_size;

    //feature enhancement
    bool use_exponential_time_gap;
    bool use_rl_cache_features;

    //RL Cache related features
    double frequency;
    uint64_t temporal_recency;
    double rho_j; //exponential smoothing of rj - temporal_recency
    uint64_t ordinal_recency;
    double delta_j;//exponential smoothing of dj - ordinal_recency

    uint64_t request_no; // request no in the trace (helps in calculating ordinal recency)
    uint64_t times_requested; //no. of times requested (helps in calculating frequency)
    std::vector<double> temporal_recency_list; //used for calculating rho_j
    std::vector<double> ordinal_recency_list; //used for calculating delta_j

    LFOFeature() {}

    LFOFeature(IdType _id, uint64_t _size, uint64_t _time)
            : id(_id),
              size(_size),
              timestamp(_time)
    {}

    ~LFOFeature() {}

    std::vector<double> get_vector() {
        std::vector<double> features;
        features.push_back(size);
        // features.push_back((optimizationGoal == BYTE_HIT_RATIO)? 1 : size);
        features.push_back(available_cache_size);

        if(use_rl_cache_features){
            features.push_back(frequency);
            features.push_back(temporal_recency);
            features.push_back(ordinal_recency);
            features.push_back(rho_j);
            features.push_back(delta_j);
            features.push_back((frequency / size));
            features.push_back((frequency * size));
        }

        auto timegap_features = get_time_gaps();
        features.insert(features.end(), timegap_features.begin(), timegap_features.end());

        return features;
    }

    std::vector<double> get_time_gaps() {
        std::vector<double> result;

        if(use_exponential_time_gap){
            std::vector<uint64_t> tmpTimeGaps;
            for(int i=1; i < 65 && i < timegaps.size(); i*=2){
                tmpTimeGaps.push_back(timegaps.at(i-1));
            }

//            1,2,4,16,32,64,128,256,512,1024,2048 - size 11
            for(int i=tmpTimeGaps.size();i<12;i++){
                result.push_back(0);//MISSING TIME GAPS
            }

            result.insert(result.end(), tmpTimeGaps.begin(), tmpTimeGaps.end());

        }else{
            for(int i=timegaps.size();i<50;i++)
                result.push_back(0); //MISSING TIME GAPS

            for (auto it = timegaps.begin(); it != timegaps.end(); ++it){
                result.push_back(*it);
            }
        }

        return result;
    }

    void calculateRhoJ(){
        alglib::real_1d_array x;// = str;
        x.setcontent(temporal_recency_list.size(), &(temporal_recency_list[0]));
        filterema(x, ALPHA);
        if(temporal_recency_list.size() > 0){
            rho_j = x[temporal_recency_list.size() - 1];
//            if(temporal_recency_list.size() > 1){
//                std::cout << "here" << std::endl;
//            }
        }
        else rho_j = 0;

    }

    void calculateDeltaJ(){
        alglib::real_1d_array x;// = str;
        x.setcontent(ordinal_recency_list.size(), &(ordinal_recency_list[0]));
        filterema(x, ALPHA);
        if(ordinal_recency_list.size() > 0) {
            delta_j = x[ordinal_recency_list.size() -1 ];
        }
        else delta_j = 0;
    }
};

#endif //WEBCDN_LFO_FEATURES_H
