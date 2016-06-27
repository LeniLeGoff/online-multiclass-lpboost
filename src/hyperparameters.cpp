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

#include <iostream>
#include <libconfig.h++>

#include "hyperparameters.h"

using namespace std;
using namespace libconfig;

using namespace oml;

void load_config_file(const string& confFile) {
    cout << "Loading config file: " << confFile << " ... ";

    Config configFile;
    configFile.readFile(confFile.c_str());

    int tmp;

    // Forest
    Hyperparameters::maxDepth = configFile.lookup("Forest.maxDepth");
    Hyperparameters::numRandomTests = configFile.lookup("Forest.numRandomTests");
    Hyperparameters::counterThreshold = configFile.lookup("Forest.counterThreshold");
    Hyperparameters::numTrees = configFile.lookup("Forest.numTrees");

    // LaRank
    Hyperparameters::larankC = configFile.lookup("LaRank.larankC");

    // Boosting
    Hyperparameters::numBases = configFile.lookup("Boosting.numBases");
    tmp = configFile.lookup("Boosting.weakLearner");
    Hyperparameters::weakLearner = (WEAK_LEARNER) tmp;

    // Online MCBoost
    Hyperparameters::shrinkage = configFile.lookup("Boosting.shrinkage");
    tmp = configFile.lookup("Boosting.lossFunction");
    Hyperparameters::lossFunction = (LOSS_FUNCTION) tmp;

    // Online MCLPBoost
    Hyperparameters::C = configFile.lookup("Boosting.C");
    Hyperparameters::cacheSize = configFile.lookup("Boosting.cacheSize");
    Hyperparameters::nuD = configFile.lookup("Boosting.nuD");
    Hyperparameters::nuP = configFile.lookup("Boosting.nuP");
    Hyperparameters::theta = configFile.lookup("Boosting.theta");
    Hyperparameters::annealingRate = configFile.lookup("Boosting.annealingRate");
    Hyperparameters::numIterations = configFile.lookup("Boosting.numIterations");

    // Experimenter
    Hyperparameters::findTrainError = configFile.lookup("Experimenter.findTrainError");
    Hyperparameters::numEpochs = configFile.lookup("Experimenter.numEpochs");

    // Data
    Hyperparameters::trainData = (const char *) configFile.lookup("Data.trainData");
    Hyperparameters::trainLabels = (const char *) configFile.lookup("Data.trainLabels");
    Hyperparameters::testData = (const char *) configFile.lookup("Data.testData");
    Hyperparameters::testLabels = (const char *) configFile.lookup("Data.testLabels");

    // Output
    Hyperparameters::savePath = (const char *) configFile.lookup("Output.savePath");
    Hyperparameters::verbose = configFile.lookup("Output.verbose");

    cout << "Done." << endl;
}
