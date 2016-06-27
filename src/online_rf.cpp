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
    m_numClasses(&numClasses), m_trueCount(0.0), m_falseCount(0.0),
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
        for (int nClass = 0; nClass < *m_numClasses; nClass++) {
            p = m_trueStats[nClass] / m_trueCount;
            trueScore += p * (1 - p);
        }
    }
        
    if (m_falseCount) {
        for (int nClass = 0; nClass < *m_numClasses; nClass++) {
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


    
template <typename hp_t>
inline void OnlineNode<hp_t>::update(const Sample& sample) {
    m_counter += sample.w;
    m_labelStats(sample.y) += sample.w;

    if (m_isLeaf) {
        // Update online tests
        for (vector<RandomTest*>::iterator itr = m_onlineTests.begin(); itr != m_onlineTests.end(); ++itr) {
            (*itr)->update(sample);
        }

        // Update the label
        m_labelStats.maxCoeff(&m_label);

        // Decide for split
        if (shouldISplit()) {
            m_isLeaf = false;

            // Find the best online test
            int nTest = 0, minIndex = 0;
            double minScore = 1, score;
            for (vector<RandomTest*>::const_iterator itr(m_onlineTests.begin()); itr != m_onlineTests.end(); ++itr, nTest++) {
                score = (*itr)->score();
                if (score < minScore) {
                    minScore = score;
                    minIndex = nTest;
                }
            }
            m_bestTest = m_onlineTests[minIndex];
            for (int nTest = 0; nTest < hp_t::numRandomTests; nTest++) {
                if (minIndex != nTest) {
                    delete m_onlineTests[nTest];
                }
            }

            // Split
            pair<VectorXd, VectorXd> parentStats = m_bestTest->getStats();
            m_rightChildNode = new OnlineNode<hp_t>(*m_numClasses, m_minFeatRange->rows(), *m_minFeatRange, *m_maxFeatRange, m_depth + 1,
                                              parentStats.first);
            m_leftChildNode = new OnlineNode<hp_t>(*m_numClasses, m_minFeatRange->rows(), *m_minFeatRange, *m_maxFeatRange, m_depth + 1,
                                             parentStats.second);
        }
    } else {
        if (m_bestTest->eval(sample)) {
            m_rightChildNode->update(sample);
        } else {
            m_leftChildNode->update(sample);
        }
    }
}

template <typename hp_t>
inline void OnlineNode<hp_t>::eval(const Sample& sample, Result& result) {
    if (m_isLeaf) {
        if (m_counter + m_parentCounter) {
            result.confidence = m_labelStats / (m_counter + m_parentCounter);
            result.prediction = m_label;
        } else {
            result.confidence = VectorXd::Constant(m_labelStats.rows(), 1.0 / *m_numClasses);
            result.prediction = 0;
        }
    } else {
        if (m_bestTest->eval(sample)) {
            m_rightChildNode->eval(sample, result);
        } else {
            m_leftChildNode->eval(sample, result);
        }
    }
}

template <typename hp_t>
inline bool OnlineNode<hp_t>::shouldISplit() const {
    bool isPure = false;
    for (int nClass = 0; nClass < *m_numClasses; nClass++) {
        if (m_labelStats(nClass) == m_counter + m_parentCounter) {
            isPure = true;
            break;
        }
    }

    if ((isPure) || (m_depth >= hp_t::maxDepth) || (m_counter < hp_t::counterThreshold)) {
        return false;
    } else {
        return true;
    }
}


template <typename hp_t>
inline void OnlineTree<hp_t>::update(Sample& sample) {
    m_rootNode->update(sample);
}

template <typename hp_t>
inline void OnlineTree<hp_t>::eval(Sample& sample, Result& result) {
    m_rootNode->eval(sample, result);
}

template <typename hp_t>
inline void OnlineRF<hp_t>::update(Sample& sample) {
    m_counter += sample.w;

    Result result(*m_numClasses), treeResult;

    int numTries;
    for (int nTree = 0; nTree < hp_t::numTrees; nTree++) {
        numTries = poisson(1.0);
        if (numTries) {
            for (int nTry = 0; nTry < numTries; nTry++) {
                m_trees[nTree]->update(sample);
            }
        } else {
            m_trees[nTree]->eval(sample, treeResult);
            result.confidence += treeResult.confidence;
        }
    }

    int pre;
    result.confidence.maxCoeff(&pre);
    if (pre != sample.y) {
        m_oobe += sample.w;
    }
}

template <typename hp_t>
inline void OnlineRF<hp_t>::eval(Sample& sample, Result& result) {
    Result treeResult;
    for (int nTree = 0; nTree < hp_t::numTrees; nTree++) {
        m_trees[nTree]->eval(sample, treeResult);
        result.confidence += treeResult.confidence;
    }

    result.confidence /= hp_t::numTrees;
    result.confidence.maxCoeff(&result.prediction);
}
