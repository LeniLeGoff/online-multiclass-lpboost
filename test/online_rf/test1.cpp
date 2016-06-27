#include <iostream>
#include <online-machine-learning/online_rf.h>
#include <online-machine-learning/hyperparameters.h>
//#include <online-machine-learning/experimenter.h>
#include <random>
#include <memory>
#include <Eigen/Core>
#include <math.h>
#include <boost/random.hpp>

#include <SFML/Graphics.hpp>

#define DIM 10000
#define BATCH 1000

void randPermute(const int& inNum, const int& inPart, vector<int>& outVect) {
    size_t size = inNum-inPart;
    outVect.resize(size);
    int randIndex, tempIndex;
    for (int nFeat = 1; nFeat <= size; nFeat++) {
        outVect[nFeat] = inNum - nFeat;
    }
    for (/*register*/ int nFeat = 0; nFeat < size; nFeat++) {
        randIndex = (int) floor(((double) size - nFeat) * randDouble()) + nFeat;
        if (randIndex == inPart) {
            randIndex--;
        }
        tempIndex = outVect[nFeat];
        outVect[nFeat] = outVect[randIndex];
        outVect[randIndex] = tempIndex;
    }

//    outVect.erase(outVect.begin() , outVect.end()+ inPart);
}


void train(Classifier* model, DataSet& dataset, Hyperparameters& hp, int sampRatio) {
    timeval startTime;
    gettimeofday(&startTime, NULL);

    std::vector<int> randIndex;

    std::vector<double> trainError(hp.numEpochs, 0.0);
    for (int nEpoch = 0; nEpoch < hp.numEpochs; nEpoch++) {
//        randPermute(dataset.m_numSamples,dataset.m_numSamples-nbr_tot_samples, randIndex);
        randPerm(dataset.m_numSamples,randIndex);
        for (int nSamp = 0; nSamp < dataset.m_numSamples; nSamp++) {
            if (hp.findTrainError) {
                Result result(dataset.m_numClasses);
                model->eval(dataset.m_samples[randIndex[nSamp]], result);
                if (result.prediction != dataset.m_samples[randIndex[nSamp]].y) {
                    trainError[nEpoch] = trainError[nEpoch] + 1.;
                }
            }
//            std::cout << " " <<  randIndex[nSamp] << std::endl;
//            Sample sam = dataset.m_samples[randIndex[nSamp]];
            model->update(dataset.m_samples[randIndex[nSamp]]);
            if (hp.verbose && (nSamp % sampRatio) == 0) {
                cout << "--- " << model->name() << " training --- Epoch: " << nEpoch + 1 << " --- ";
                cout << (10 * nSamp) / sampRatio << "%";
                cout << " --- Training error = " << trainError[nEpoch] << "/" << nSamp << endl;
            }
        }
    }

    timeval endTime;
    gettimeofday(&endTime, NULL);
    std::cout << "--- " << model->name() << " training time = ";
    std::cout << (endTime.tv_sec - startTime.tv_sec + (endTime.tv_usec - startTime.tv_usec) / 1e6) << " seconds." << std::endl;
}

float test(Classifier *model){
//    std::cout << "generate test set class 0" << std::endl;
    float cpt = 0;
    for(int i = 0; i < 10; i++){
        Sample sample;
        sample.id = rand();
        Vector2d x;
        x(0) = 20*cos((double)(rand()%628)/100.) + 50;
        x(1) = 20*sin((double)(rand()%628)/100.) + 50;
        sample.x = x;
        sample.y = 0;
        sample.w = 0.1;

        Result r(2);
        model->eval(sample,r);

        if(r.prediction != sample.y)
            cpt+=1.;

//        std::cout << r.prediction << std::endl;
    }

//    std::cout << "generate test set class 1" << std::endl;
    int alt = 1;
    for(int i = 0; i < 10; i++){
        Sample sample;
        sample.id = rand();
        Vector2d x;
        double theta = rand()%628;

        x(0) = 50.*cos((double)(theta)/100.) + 50;
        x(1) = 50.*sin((double)(theta)/100.) + 50;
        sample.x = x;
        sample.y = 1;
        sample.w = 0.1;

        Result r(2);
        model->eval(sample,r);
        if(r.prediction != sample.y)
            cpt+=1.;
//        std::cout << r.prediction << std::endl;
    }

    return cpt/20.;
}

void generate_train_dataset(std::vector<Sample>& samples, std::vector<sf::CircleShape>& circles,const std::vector<Vector2d>& space){
    double A1 = rand()%201/100. - 1.;
    double B1 = rand()%201/100. - 1.;
    double A2 = rand()%201/100. - 1.;
    double B2 = rand()%201/100. - 1.;
    double A11 = rand()%201/100. - 1.;
    double B11 = rand()%201/100. - 1.;
    double A21 = rand()%201/100. - 1.;
    double B21 = rand()%201/100. - 1.;

    double c_x1 = rand()%21 + 40;
    double c_y1 = rand()%21 + 40;
    double c_x2 = rand()%21 + 40;
    double c_y2 = rand()%21 + 40;


    double c_x11 = rand()%21 + 40;
    double c_y11 = rand()%21 + 40;
    double c_x21 = rand()%21 + 40;
    double c_y21 = rand()%21 + 40;


    double s_x1 = rand()%30 + 10;
    double s_y1 = rand()%30;
    double s_x2 = rand()%30 + 10;
    double s_y2 = rand()%30;


    double s_x11 = rand()%30 + 10;
    double s_y11 = rand()%30;
    double s_x21 = rand()%30 + 10;
    double s_y21 = rand()%30;

    int alt = 1;

    for(int i = 0; i < samples.size()/2; i++){
        samples[i].id = rand();
        Vector2d x;
//        double A = 1., B = 1.;

        if(alt){
            double theta = rand()%628/100.;
            double radius = (rand()%(int)(space[1][0]-s_x11)+s_y11);
            double den = A11*cos(theta) + B11*sin(theta) + 1;
            x(0) = radius*cos(theta)/den + c_x11;
            x(1) = radius*sin(theta)/den + c_y11;
//            alt = 0;
        }
        else {

            double theta = rand()%628/100.;
            double radius = (rand()%(int)(space[1][0]-s_x1)+s_y1);
            double den = A1*cos(theta) + B1*sin(theta) + 1;
            x(0) = radius*cos(theta)/den + c_x1;
            x(1) = radius*sin(theta)/den + c_y1;
            alt = 1;
        }
        samples[i].x = x;
        samples[i].y = 0;
        samples[i].w = 0.1;
        circles[i].setPosition(samples[i].x(0)*4,samples[i].x(1)*4);
        circles[i].setFillColor(sf::Color::Blue);
    }
    for(int i = samples.size()/2; i < samples.size(); i++){
        samples[i].id = rand();
        Vector2d x;
//        double A = 1., B = 1.;
        if(alt){
            double theta = rand()%628/100.;
            double radius = (rand()%(int)(space[1][0]-s_x21)+s_y21);
            double den = A21*cos(theta) + B21*sin(theta) + 1;
            x(0) = radius*cos(theta)/den + c_x21;
            x(1) = radius*sin(theta)/den + c_y21;
//            alt = 0;
        }
        else {

            double theta = rand()%628/100.;
            double radius = (rand()%(int)(space[1][0]-s_x2)+s_y2);
            double den = A2*cos(theta) + B2*sin(theta) + 1;
            x(0) = radius*cos(theta)/den + c_x2;
            x(1) = radius*sin(theta)/den + c_y2;
            alt = 1;
        }
//        if(alt){
//            double theta = rand()%628;
//            double radius = (rand()%(int)(space[1][0]-80)+50);
//            x(0) = radius*cos((double)(theta)/100.) + space[1][0]/2;
//            x(1) = radius*sin((double)(theta)/100.) + space[1][0]/2;
//            alt = 0;

//        }else {

//        double theta = rand()%628;
//        double radius = (rand()%(int)(space[1][0]-80));
//            x(0) = 2*radius*cos((double)(theta)/100.) + space[1][0]/2;
//            x(1) = radius*sin((double)(theta)/100.) + space[1][0]/2;
//            alt = 1;
//        }
        samples[i].x = x;
        samples[i].y = 1;
        samples[i].w = 0.1;

        circles[i].setPosition(samples[i].x(0)*4,samples[i].x(1)*4);

        circles[i].setFillColor(sf::Color::Red);
    }

}

void minIndexes(const VectorXd& vect, std::vector<int>& indexes){
    int minVal = vect(0);
    int min_i = 0;
    for(int i = 1; i < vect.size(); i++){
        if(minVal > vect(i)){
            minVal = vect(i);
            min_i = i;
        }
    }
    indexes.push_back(min_i);
    for(int i = 0; i < vect.size(); i++){
        if(minVal >= vect(i))
            indexes.push_back(i);
    }

}

int main(int argc, char **argv){
    srand(std::time(NULL));
    std::vector<Vector2d> space = {Vector2d(0,0),Vector2d(100,100)};

    std::cout << "generating dataset" << std::endl;
    DataSet dataset;
    std::vector<Sample> d_samples(space[1][0]*space[1][1]);
    std::vector<Sample> samples(DIM);
    std::vector<sf::CircleShape> circles(DIM,sf::CircleShape(3));
    std::vector<sf::RectangleShape> rects_fuzzy(space[1][0]*space[1][1],sf::RectangleShape(sf::Vector2f(4,4)));
    std::vector<sf::RectangleShape> rects_exact(space[1][0]*space[1][1],sf::RectangleShape(sf::Vector2f(4,4)));


    Vector2d x(0,0);
    for(int i = 0; i < space[1][0]*space[1][1]; i++){
        Vector2d x;
        x(0) = i%(int)(space[1][0]);
        if(x(0) == space[1][0]-1)
            x(1)++;
        rects_fuzzy[i].setPosition(x(0)*4+space[1][1]*4,x(1)*4);
        rects_fuzzy[i].setFillColor(sf::Color::White);
        rects_exact[i].setPosition(x(0)*4+space[1][1]*4*2,x(1)*4);
        rects_exact[i].setFillColor(sf::Color::White);

        d_samples[i].id = rand();
        d_samples[i].x = x;
        d_samples[i].y = 0;
        d_samples[i].w = 0.1;
    }

    generate_train_dataset(samples,circles,space);

    //    dataset.m_samples = samples;
    dataset.m_minFeatRange = space[0];
    dataset.m_maxFeatRange = space[1];
    dataset.m_numClasses = 2;
    dataset.m_numFeatures = 2;
    dataset.m_numSamples = BATCH;
    dataset.m_samples.resize(BATCH);


    Hyperparameters hp;//(/*"/home/lenilegoff/project_test/online_rf/test.conf"*/);


    hp.numRandomTests = 10;
    hp.maxDepth = 10;
    hp.counterThreshold = 200;
    hp.numTrees = 10;
    hp.numEpochs = 1;
    hp.findTrainError = 0;

    //    hp.trainData = "/home/leni/git/online-multiclass-lpboost/data/dna-train.data";
    //    hp.trainLabels = "/home/leni/git/online-multiclass-lpboost/data/dna-train.labels";

    hp.savePath = "~/test_projects/";
    hp.verbose = 0;

    //    dataset.load(hp.trainData,hp.trainLabels);

    OnlineRF* online_rf = new OnlineRF(hp,dataset.m_numClasses,dataset.m_numFeatures,dataset.m_minFeatRange,dataset.m_maxFeatRange);
    std::vector<int> r_indexes;

    int size = DIM;
    r_indexes.push_back(rand()%size);
    int c_index = 0;
    dataset.m_samples[c_index] = samples[r_indexes.back()];
    size--;
    c_index++;
    dataset.m_numSamples = 1;
    samples.erase(samples.begin()+r_indexes.back());
    samples.resize(size);
    std::vector<double> confidence_matrix(space[1][0]*space[1][1],0.);



    std::vector<Result> res(space[1][0]*space[1][1],Result(2));

    sf::RenderWindow window(sf::VideoMode(space[1][0]*4*3,space[1][1]*4),"dataset");
    while(window.isOpen()){

        sf::Event event;
        while (window.pollEvent(event))
        {
            if (event.type == sf::Event::Closed)
                window.close();
        }

        window.clear(sf::Color::White);

        for(int i = 0; i < r_indexes.size(); i++)
            window.draw(circles[r_indexes[i]]);
        for(auto rect : rects_fuzzy)
            window.draw(rect);

        for(auto rect : rects_exact)
            window.draw(rect);

        if(size>0){
            r_indexes.push_back(rand()%size);
            dataset.m_samples[c_index] = samples[r_indexes.back()];
            size--;
            c_index++;
            dataset.m_numSamples++;
            samples.erase(samples.begin()+r_indexes.back());
            samples.resize(size);
//            confidence_matrix.erase(confidence_matrix.begin()+r_indexes.back());
//            confidence_matrix.resize(size);

            if(c_index >= BATCH)
                c_index = 0;

            if(dataset.m_numSamples > BATCH)
                dataset.m_numSamples = BATCH;


            std::cout << "with " << (DIM-size);
            train(online_rf,dataset,hp,0.5);


            for(int i = 0; i < d_samples.size(); i++){
                    online_rf->eval(d_samples[i],res[i]);
                    float c1 = res[i].confidence(0);
                    if(c1 > 1)
                        c1 = 1;
                    else if (c1<0)
                        c1 = 0;
                    float c2 = res[i].confidence(1);
                    if(c2 > 1)
                        c2 = 1;
                    else if (c2<0)
                        c2 = 0;
//                    if(!res[i].prediction)
//                        rects_fuzzy[i].setFillColor(sf::Color(c*255,c*255,c*255));
//                    else
//                        rects_fuzzy[i].setFillColor(sf::Color((res[i].confidence(0))*255,(res[i].confidence(0))*255,(res[i].confidence(0))*255));

                    confidence_matrix[i] = c1;
                    rects_fuzzy[i].setFillColor(sf::Color(((1-c1)+(c2))/2.*255,0,(c1+(1-c2))/2.*255));
                    if((c1 == 1 && res[i].prediction == 0) || (c2 == 1 && res[i].prediction == 1))
                        rects_exact[i].setFillColor(sf::Color((res[i].prediction)*255,0,(1-res[i].prediction)*255));
                    else
                        rects_exact[i].setFillColor(sf::Color(255,0,255));

            }

        }


        window.display();
    }


    //std::vector<Result> results;


//    while(size > 0){

//        index = rand()%size;
//        dataset.m_samples.push_back(samples[index]);
//        size--;
//        dataset.m_numSamples = DIM - size;
//        samples.erase(samples.begin()+index);
//        samples.resize(size);

//        train(online_rf,dataset,hp, 0.5);

//        std::cout << "with "<< (DIM-size) << " samples, test error : " << test(online_rf,dataset) << std::endl;
//    }
////    for(int i = 0; i < samples.size(); i++)
////        online_rf.update(samples[i]);




//    std::cout << "evaluation" << std::endl;
////    std::vector<Result> res = test(online_rf,dataset,hp);



    return 0;
}



