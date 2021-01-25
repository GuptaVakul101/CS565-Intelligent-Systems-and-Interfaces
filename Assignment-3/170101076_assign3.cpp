#include <bits/stdc++.h>
using namespace std;

// for counting how many times the weights increase or decrease
long long positiveDelta = 0;
long long negativeDelta = 0;

// given a set of data points -> it returns a softmax vector converting the data points into probabilities
vector<long double> getSoftmaxVector(vector<long double> inputVector){
    long long n = inputVector.size();
    // base case -> return an empty vector
    if(n == 0){
        return {};
    }
    vector<long double> result(n,0);
    // calculate the sum of all terms
    long double sum = 0;
    for(long long i = 0; i < n; i++){
        result[i] = exp(inputVector[i]);
        sum += result[i];
    }
    // convert every point into probability by dividing with sum
    for(long long i = 0; i < n; i++){
        result[i] /= sum;
    }
    return result;
}

// given an input matrix -> it returns the transpose of that matrix
vector<vector<long double>> getTransposeMatrix(vector<vector<long double>> inputMatrix){
    // base cases -> if n or m is 0, return an empty matrix
    long long n = inputMatrix.size();
    if(n == 0){
        return {{}};
    }
    long long m = inputMatrix[0].size();
    if(m == 0){
        return {{}};
    }
    vector<vector<long double>> transposedMatrix(m,vector<long double>(n,0));
    for(long long i = 0; i < m; i++){
        for(long long j = 0; j < n; j++){
            // swap i and j to get transpose
            transposedMatrix[i][j] = inputMatrix[j][i];
        }
    }
    return transposedMatrix;
}

// given a matrix and a vector, it returns the dot product of them returning a vector
vector<long double> getDotProduct(vector<vector<long double>> inputMatrix, vector<long double> inputVector){
    // base cases -> if n or m is 0, return an empty vector
    long long n = inputMatrix.size();
    if(n == 0){
        return {};
    }
    long long m = inputMatrix[0].size();
    if(m == 0){
        return {};
    }

    vector<long double> result(n,0);
    for(long long i = 0; i < n; i++){
        // get result[i] by summing over j
        for(long long j = 0; j < m; j++){
            result[i] += inputMatrix[i][j]*inputVector[j];
        }
    }
    return result;
}

// update the hidden output matrix by calculating the required gradient
void updateHiddenOutputWeightMatrix(long long numberUnitsHidden, long long wordOutput, long double rateOfLearning, vector<vector<long double>> &hiddenOutputWeightMatrix, vector<long double> &errorVector, vector<long double> &h){
    // calculating the gradient vector
    vector<long double> delE_By_DelW(numberUnitsHidden,0);
    for(long long k = 0; k < numberUnitsHidden; k++){
        delE_By_DelW[k] = errorVector[wordOutput]*h[k];
    }
    // update the hidden output matrix
    for(long long k = 0; k < numberUnitsHidden; k++){
        hiddenOutputWeightMatrix[k][wordOutput] -= rateOfLearning*delE_By_DelW[k];
        // increment the negative and positive delta counts based on sign
        if(rateOfLearning*delE_By_DelW[k] > 0){
            negativeDelta++;
        }
        else{
            positiveDelta++;
        }
    }
}

// update the input hidden matrix by calculating the required gradient
void updateInputHiddenWeightMatrix(long long numberUnitsHidden, long long sizeOfVocabulary, long long wordInput, long double rateOfLearning, vector<vector<long double>> &hiddenOutputWeightMatrix, vector<long double> &errorVector, vector<vector<long double>> &inputHiddenWeightMatrix){
    // calculating the gradient vector
    vector<long double> delE_By_DelH(numberUnitsHidden,0);
    for(long long k = 0; k < numberUnitsHidden; k++){
        for(long long l = 0; l < sizeOfVocabulary; l++){
            delE_By_DelH[k] += errorVector[l]*hiddenOutputWeightMatrix[k][l];
        }
    }
    // update the input hidden matrix
    for(long long l = 0; l < numberUnitsHidden; l++){
        inputHiddenWeightMatrix[wordInput][l] -= rateOfLearning*delE_By_DelH[l];
        // increment the negative and positive delta counts based on sign
        if(rateOfLearning*delE_By_DelH[l] > 0){
            negativeDelta++;
        }
        else{
            positiveDelta++;
        }
    }
}

// main function -> execution starts from here
int main(){

    // taking the input in the specified format
    long long sizeOfVocabulary, numberUnitsHidden;
    long double rateOfLearning;
    long long numberOfIterations, numberOfPairs;
    cin >> sizeOfVocabulary >> numberUnitsHidden >> rateOfLearning >> numberOfIterations >> numberOfPairs;
    vector<pair<long double,long double>> word_pairs;
    long long iterationNumber, wordInput, wordOutput;
    for(long long i = 0; i < numberOfPairs; i++){
        cin >> iterationNumber >> wordInput >> wordOutput;
        word_pairs.push_back(make_pair(wordInput-1,wordOutput-1));
    }

    // initialise both the weight matrices with appropriate dimensions and with the value 0.5
    vector<vector<long double>> inputHiddenWeightMatrix(sizeOfVocabulary,vector<long double>(numberUnitsHidden,0.5));
    vector<vector<long double>> hiddenOutputWeightMatrix(numberUnitsHidden,vector<long double>(sizeOfVocabulary,0.5));

    // running all the iterations
    for(long long i = 0; i < numberOfIterations; i++){
        // do this task for each input-output pair
        for(long long j = 0; j < numberOfPairs; j++){

            // initialising these every time
            positiveDelta = 0;
            negativeDelta = 0;

            // obtain the indices of input and output word
            wordInput = word_pairs[j].first;
            wordOutput = word_pairs[j].second;

            // form the input one hot encoding by setting at the input word index
            vector<long double> inputOneHotEncoding(sizeOfVocabulary,0);
            inputOneHotEncoding[wordInput] = 1;

            // calculating intermediate vector by taking the dot product of the transposed weight matrix with input vector
            vector<long double> h = getDotProduct(getTransposeMatrix(inputHiddenWeightMatrix),inputOneHotEncoding);
            // calculating the output vector
            vector<long double> outputVector = getDotProduct(getTransposeMatrix(hiddenOutputWeightMatrix),h);
            // taking softmax of the output vector
            vector<long double> outputSoftmaxVector = getSoftmaxVector(outputVector);
            // defining the error vector
            vector<long double> errorVector = outputSoftmaxVector;
            errorVector[wordOutput]--;

            // update the input hidden matrix by calculating the required gradient
            updateInputHiddenWeightMatrix(numberUnitsHidden, sizeOfVocabulary, wordInput, rateOfLearning, hiddenOutputWeightMatrix, errorVector, inputHiddenWeightMatrix);
            // update the hidden output matrix by calculating the required gradient
            updateHiddenOutputWeightMatrix(numberUnitsHidden, wordOutput, rateOfLearning, hiddenOutputWeightMatrix, errorVector, h);

            // printing the result in desired format
            cout << (i+1) << " " << (j+1) << " " << negativeDelta << " " << positiveDelta << endl;
        }
    }
}
