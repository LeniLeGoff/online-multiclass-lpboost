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

#include <fstream>
#include <sys/time.h>

#include "experimenter.h"

using namespace oml;



double experimenter::compError(const vector<Result>& results, const DataSet& dataset) {
    double error = 0.0;
    for (int nSamp = 0; nSamp < dataset.m_numSamples; nSamp++) {
        if (results[nSamp].prediction != dataset.m_samples[nSamp].y) {
            error++;
        }
    }

    return error / dataset.m_numSamples;
}

void experimenter::dispErrors(const vector<double>& errors) {
    for (int nSamp = 0; nSamp < (int) errors.size(); nSamp++) {
        cout << nSamp + 1 << ": " << errors[nSamp] << " --- ";
    }
    cout << endl;
}
