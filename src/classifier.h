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

#ifndef CLASSIFIER_H_
#define CLASSIFIER_H_

#include "data.h"
#include "hyperparameters.h"
#include <memory>
#include <boost/serialization/access.hpp>
#include <boost/serialization/string.hpp>


using namespace std;

namespace oml{

class Classifier {
public:

    friend class boost::serialization::access;

    Classifier(){}
    Classifier(const int& numClasses) :
        m_numClasses(numClasses) {
    }
    Classifier(const Hyperparameters &hp, const int &numClasses) :
       m_hp(&hp), m_numClasses(numClasses) {
    }
    Classifier(const Classifier& c)
        : m_numClasses(c.m_numClasses), m_name(c.m_name), m_hp(c.m_hp){}

    virtual ~Classifier(){

    }

    virtual void update(Sample& sample) = 0;
    virtual void eval(Sample& sample, Result& result) = 0;

    const string name() const {
        return m_name;
    }

    template <typename archive>
    void serialize(archive &arch,const unsigned int v){
        arch & m_numClasses;
        arch & m_name;
    }

protected:
    int m_numClasses;
    const Hyperparameters* m_hp;
    string m_name;
};
}

#endif /* CLASSIFIER_H_ */
