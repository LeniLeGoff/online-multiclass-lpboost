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

#include <online_rf.h>

using namespace oml;

RandomTest::RandomTest(const int& numClasses, const int& numFeatures, const VectorXd &minFeatRange, const VectorXd &maxFeatRange) :
    m_numClasses(numClasses), m_trueCount(0.0), m_falseCount(0.0),
    m_trueStats(VectorXd::Zero(numClasses)), m_falseStats(VectorXd::Zero(numClasses)) {
    m_feature = randDouble(0, numFeatures /*+ 1*/);
    m_threshold = randDouble(minFeatRange(m_feature), maxFeatRange(m_feature));
}

void RandomTest::update(const Sample& sample) {
    updateStats(sample, eval(sample));
}
    
bool RandomTest::eval(const Sample& sample) const {
    return (sample.x(m_feature) > m_threshold) ? true : false;
}
    
double RandomTest::score() const {
    double trueScore = 0.0, falseScore = 0.0, p;
    if (m_trueCount) {
        for (int nClass = 0; nClass < m_numClasses; nClass++) {
            p = m_trueStats[nClass] / m_trueCount;
            trueScore += p * (1 - p);
        }
    }
        
    if (m_falseCount) {
        for (int nClass = 0; nClass < m_numClasses; nClass++) {
            p = m_falseStats[nClass] / m_falseCount;
            falseScore += p * (1 - p);
        }
    }
        
    return (m_trueCount * trueScore + m_falseCount * falseScore) / (m_trueCount + m_falseCount + 1e-16);
}
    
pair<VectorXd, VectorXd > RandomTest::getStats() const {
    return pair<VectorXd, VectorXd> (m_trueStats, m_falseStats);
}

void RandomTest::updateStats(const Sample& sample, const bool& decision) {
    if (decision) {
        m_trueCount += sample.w;
        m_trueStats(sample.y) += sample.w;
    } else {
        m_falseCount += sample.w;
        m_falseStats(sample.y) += sample.w;
    }
}    


    

