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

#ifndef HYPERPARAMETERS_H_
#define HYPERPARAMETERS_H_

#include <string>
using namespace std;

namespace oml{

typedef enum {
    EXPONENTIAL, LOGIT
} LOSS_FUNCTION;

typedef enum {
    WEAK_ORF, WEAK_LARANK
} WEAK_LEARNER;

struct Hyperparameters {
    // Forest
    static int numRandomTests;
    static int counterThreshold;
    static int maxDepth;
    static int numTrees;

    // Linear LaRank
    static double larankC;

    // Boosting
    static int numBases;
    static WEAK_LEARNER weakLearner;

    // Online MCBoost
    static double shrinkage;
    static LOSS_FUNCTION lossFunction;

    // Online MCLPBoost
    static double C;
    static int cacheSize;
    static double nuD;
    static double nuP;
    static double annealingRate;
    static double theta;
    static  int numIterations;

    // Experimenter
    static int findTrainError;
    static int numEpochs;

    // Data
    static string trainData;
    static string trainLabels;
    static string testData;
    static string testLabels;

    // Output
    static string savePath;
    static int verbose;
};

void load_config_file(const string& confFile);

}


#endif /* HYPERPARAMETERS_H_ */
