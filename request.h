#ifndef REQUEST_H
#define REQUEST_H

#include <cstdint>
#include <iostream>
#include <lib/dlib/dlib/matrix.h>

typedef uint64_t IdType;

// Request information
class SimpleRequest
{
private:
    IdType _id; // request object id
    uint64_t _size; // request size in bytes
    uint64_t _timestamp; // request timestamp
    std::vector<double> _featureVector;//feature vector
    dlib::matrix<double, 0, 1> _sample;

public:
    SimpleRequest()
    {
    }
    virtual ~SimpleRequest()
    {
    }

    // Create request
    SimpleRequest(IdType id, uint64_t size)
        : _id(id),
          _size(size)
    {
    }

    SimpleRequest(IdType id, uint64_t size, uint64_t time)
        : _id(id),
          _size(size),
          _timestamp(time)
    {}

    void reinit(IdType id, uint64_t size)
    {
        _id = id;
        _size = size;
    }

    void reinit(IdType id, uint64_t size, uint64_t time) {
        _id = id;
        _size = size;
        _timestamp = time;
    }

    // Print request to stdout
    void print() const
    {
        std::cout << "id" << getId() << " size " << getSize() << std::endl;
    }

    // Get request object id
    IdType getId() const
    {
        return _id;
    }

    // Get request size in bytes
    uint64_t getSize() const
    {
        return _size;
    }

    uint64_t getTimeStamp() const
    {
        return _timestamp;
    }

    const std::vector<double> &getFeatureVector() const {
        return _featureVector;
    }

    void setFeatureVector(const std::vector<double> &featureVector) {
        _featureVector = featureVector;
    }

    void setSampleType(dlib::matrix<double, 0, 1> &sampleType) {
        _sample = sampleType;
    }

    const dlib::matrix<double, 0, 1> &getSampleType() const {
        return _sample;
    }
};


#endif /* REQUEST_H */



