// -*- C++ -*-
/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2010 Amir Saffari, amir@ymer.org
 * Copyright (C) 2010 Amir Saffari,
 *                    Institute for Computer Graphics and Vision,
 *                    Graz University of Technology, Austria
 */

#ifndef EXPERIMENTER_H
#define EXPERIMENTER_H

#include "classifier.h"
#include "data.h"
#include "utilities.h"

namespace oml{

class experimenter{
public:
    template <typename hp_t>
    static void train(std::shared_ptr<Classifier> model, DataSet& dataset);

    template <typename hp_t>
    static vector<Result> test(std::shared_ptr<Classifier> model, DataSet& dataset);

    template <typename hp_t>
    static vector<Result> trainAndTest(std::shared_ptr<Classifier> model, DataSet& dataset_tr, DataSet& dataset_ts);

    static double compError(const vector<Result>& results, const DataSet& dataset);
    static void dispErrors(const vector<double>& errors);

};

template <typename hp_t>
void experimenter::train(std::shared_ptr<Classifier> model, DataSet& dataset) {
    timeval startTime;
    gettimeofday(&startTime, NULL);

    vector<int> randIndex;
    int sampRatio = dataset.m_numSamples / 10;
    vector<double> trainError(hp_t::numEpochs, 0.0);
    for (int nEpoch = 0; nEpoch < hp_t::numEpochs; nEpoch++) {
        randPerm(dataset.m_numSamples, randIndex);
        for (int nSamp = 0; nSamp < dataset.m_numSamples; nSamp++) {
            if (hp_t::findTrainError) {
                Result result(dataset.m_numClasses);
                model->eval(dataset.m_samples[randIndex[nSamp]], result);
                if (result.prediction != dataset.m_samples[randIndex[nSamp]].y) {
                    trainError[nEpoch] = trainError[nEpoch] + 1.;
                }
            }

            model->update(dataset.m_samples[randIndex[nSamp]]);
            if (hp_t::verbose && (nSamp % sampRatio) == 0) {
                cout << "--- " << model->name() << " training --- Epoch: " << nEpoch + 1 << " --- ";
                cout << (10 * nSamp) / sampRatio << "%";
                cout << " --- Training error = " << trainError[nEpoch] << "/" << nSamp << endl;
            }
        }
    }

    timeval endTime;
    gettimeofday(&endTime, NULL);
    cout << "--- " << model->name() << " training time = ";
    cout << (endTime.tv_sec - startTime.tv_sec + (endTime.tv_usec - startTime.tv_usec) / 1e6) << " seconds." << endl;
}

template <typename hp_t>
vector<Result> experimenter::test(std::shared_ptr<Classifier> model, DataSet& dataset) {
    timeval startTime;
    gettimeofday(&startTime, NULL);

    vector<Result> results;
    for (int nSamp = 0; nSamp < dataset.m_numSamples; nSamp++) {
        Result result(dataset.m_numClasses);
        model->eval(dataset.m_samples[nSamp], result);
        results.push_back(result);
    }

    double error = compError(results, dataset);
    if (hp_t::verbose) {
        cout << "--- " << model->name() << " test error: " << error << endl;
    }

    timeval endTime;
    gettimeofday(&endTime, NULL);
    cout << "--- " << model->name() << " testing time = ";
    cout << (endTime.tv_sec - startTime.tv_sec + (endTime.tv_usec - startTime.tv_usec) / 1e6) << " seconds." << endl;

    return results;
}

template <typename hp_t>
vector<Result> experimenter::trainAndTest(std::shared_ptr<Classifier> model, DataSet& dataset_tr, DataSet& dataset_ts) {
    timeval startTime;
    gettimeofday(&startTime, NULL);

    vector<Result> results;
    vector<int> randIndex;
    int sampRatio = dataset_tr.m_numSamples / 10;
    vector<double> trainError(hp_t::numEpochs, 0.0);
    vector<double> testError;
    for (int nEpoch = 0; nEpoch < hp_t::numEpochs; nEpoch++) {
        randPerm(dataset_tr.m_numSamples, randIndex);
        for (int nSamp = 0; nSamp < dataset_tr.m_numSamples; nSamp++) {
            if (hp_t::findTrainError) {
                Result result(dataset_tr.m_numClasses);
                model->eval(dataset_tr.m_samples[randIndex[nSamp]], result);
                if (result.prediction != dataset_tr.m_samples[randIndex[nSamp]].y) {
                    trainError[nEpoch]++;
                }
            }

            model->update(dataset_tr.m_samples[randIndex[nSamp]]);
            if (hp_t::verbose && (nSamp % sampRatio) == 0) {
                cout << "--- " << model->name() << " training --- Epoch: " << nEpoch + 1 << " --- ";
                cout << (10 * nSamp) / sampRatio << "%";
                cout << " --- Training error = " << trainError[nEpoch] << "/" << nSamp << endl;
            }
        }

        results = test<hp_t>(model, dataset_ts);
        testError.push_back(compError(results, dataset_ts));
    }

    timeval endTime;
    gettimeofday(&endTime, NULL);
    cout << "--- Total training and testing time = ";
    cout << (endTime.tv_sec - startTime.tv_sec + (endTime.tv_usec - startTime.tv_usec) / 1e6) << " seconds." << endl;

    if (hp_t::verbose) {
        cout << endl << "--- " << model->name() << " test error over epochs: ";
        dispErrors(testError);
    }

    // Write the results
    string saveFile = hp_t::savePath + ".errors";
    ofstream file(saveFile.c_str(), ios::binary);
    if (!file) {
        cout << "Could not access " << saveFile << endl;
        exit(EXIT_FAILURE);
    }
    file << hp_t::numEpochs << " 1" << endl;
    for (int nEpoch = 0; nEpoch < hp_t::numEpochs; nEpoch++) {
        file << testError[nEpoch] << endl;
    }
    file.close();

    return results;
}

}

#endif // EXPERIMENTER_H
