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

#ifndef ONLINERF_H_
#define ONLINERF_H_

#include "classifier.h"
#include "data.h"
#include "hyperparameters.h"
#include "utilities.h"

namespace oml {

class RandomTest {
public:
    RandomTest(const int& numClasses, const int& numFeatures, const VectorXd &minFeatRange, const VectorXd &maxFeatRange);

    void update(const Sample& sample);

    bool eval(const Sample& sample) const;

    double score() const;

    pair<VectorXd, VectorXd > getStats() const;

protected:
    const int* m_numClasses;
    int m_feature;
    double m_threshold;

    double m_trueCount;
    double m_falseCount;
    VectorXd m_trueStats;
    VectorXd m_falseStats;

    void updateStats(const Sample& sample, const bool& decision);
};

template <typename hp_t>
class OnlineNode {
public:
    OnlineNode(const int& numClasses, const int& numFeatures, const VectorXd& minFeatRange, const VectorXd& maxFeatRange,
               const int& depth) :
        m_numClasses(&numClasses), m_depth(depth), m_isLeaf(true), m_label(-1),
        m_counter(0.0), m_parentCounter(0.0), m_labelStats(VectorXd::Zero(numClasses)),
        m_minFeatRange(&minFeatRange), m_maxFeatRange(&maxFeatRange) {
        // Creating random tests
        for (int nTest = 0; nTest < hp_t::numRandomTests; nTest++) {
            m_onlineTests.push_back(new RandomTest(numClasses, numFeatures, minFeatRange, maxFeatRange));
        }
    }

    OnlineNode(const int& numClasses, const int& numFeatures, const VectorXd& minFeatRange, const VectorXd& maxFeatRange,
               const int& depth, const VectorXd& parentStats):
        m_numClasses(&numClasses), m_depth(depth), m_isLeaf(true), m_label(-1),
        m_counter(0.0), m_parentCounter(parentStats.sum()), m_labelStats(parentStats),
        m_minFeatRange(&minFeatRange), m_maxFeatRange(&maxFeatRange) {
        m_labelStats.maxCoeff(&m_label);
        // Creating random tests
        for (int nTest = 0; nTest < hp_t::numRandomTests; nTest++) {
            m_onlineTests.push_back(new RandomTest(numClasses, numFeatures, minFeatRange, maxFeatRange));
        }
    }

    ~OnlineNode() {
        if (!m_isLeaf) {
            delete m_leftChildNode;
            delete m_rightChildNode;
            delete m_bestTest;
        } else {
            for (int nTest = 0; nTest < hp_t::numRandomTests; nTest++) {
                delete m_onlineTests[nTest];
            }
        }
    }

    void update(const Sample& sample);
    void eval(const Sample& sample, Result& result);

private:
    const int* m_numClasses;
    int m_depth;
    bool m_isLeaf;
    const Hyperparameters* m_hp;
    int m_label;
    double m_counter;
    double m_parentCounter;
    VectorXd m_labelStats;
    const VectorXd* m_minFeatRange;
    const VectorXd* m_maxFeatRange;

    OnlineNode* m_leftChildNode;
    OnlineNode* m_rightChildNode;

    vector<RandomTest*> m_onlineTests;
    RandomTest* m_bestTest;

    bool shouldISplit() const;
};

template <typename hp_t>
class OnlineTree: public Classifier {
public:
    OnlineTree(const int& numClasses,
               const int& numFeatures, const VectorXd& minFeatRange, const VectorXd& maxFeatRange)
    : Classifier(numClasses) {
        m_rootNode = new OnlineNode<hp_t>(numClasses, numFeatures, minFeatRange, maxFeatRange, 0);
        m_name = "OnlineTree";
    }

    ~OnlineTree() {
        delete m_rootNode;
    }

    virtual void update(Sample& sample);

    virtual void eval(Sample& sample, Result& result);

private:
    OnlineNode<hp_t>* m_rootNode;
};

template <typename hp_t>
class OnlineRF: public Classifier {
public:
    OnlineRF(const int& numClasses,
             const int& numFeatures, const VectorXd& minFeatRange, const VectorXd& maxFeatRange) :
        Classifier(numClasses), m_counter(0.0), m_oobe(0.0) {
        typename OnlineTree<hp_t>::Ptr tree;
        for (int nTree = 0; nTree < hp_t::numTrees; nTree++) {
            tree.reset(new OnlineTree<hp_t>( numClasses, numFeatures, minFeatRange, maxFeatRange));
            m_trees.push_back(tree);
        }
        m_name = "OnlineRF";
    }

    ~OnlineRF() {
        for (int nTree = 0; nTree < hp_t::numTrees; nTree++) {
            delete m_trees[nTree];
        }
    }

    virtual void update(Sample& sample);

    virtual void eval(Sample& sample, Result& result);

protected:
    double m_counter;
    double m_oobe;

    vector< typename OnlineTree<hp_t>::Ptr> m_trees;
};
}

#endif /* ONLINERF_H_ */
