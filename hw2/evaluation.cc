#include "cnn/nodes.h"
#include "cnn/cnn.h"
#include "cnn/training.h"
#include "cnn/timing.h"
#include "cnn/rnn.h"
#include "cnn/gru.h"
#include "cnn/lstm.h"
#include "cnn/dict.h"
#include "cnn/expr.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

using namespace std;
using namespace cnn;

unsigned GLOVE_DIM = 50;
unsigned SENTENCE_DIM = 100;
unsigned INPUT_DIM = GLOVE_DIM * GLOVE_DIM + SENTENCE_DIM;
unsigned HIDDEN_DIM = 45;
unsigned PAIRWISE_DIM = 100;
unsigned OUTPUT_DIM = 1;


class Sentence {
  public:
    vector<float> syn;
    vector<vector<float>> sem;
    float BLEU;
    float meteor;
};

class Instance {
  public:
    Sentence hyp1;
    Sentence hyp2;
    Sentence ref;
    int correct;
};


unordered_map<string, vector<float>> getAllWords(string hyp1, string hyp2, string ref, string gloveFile){
  string word; //current word read in
  string line;
  string element;
  unordered_set<string>  words; //a list (basically) of all of the words in the hypothesis and reference-make this a hash?
  unordered_map<string, vector<float>> word2gl; //a map storing all of the strings and their embeddings
  for (string file : {hyp1, hyp2, ref}) {
    //for each word in the input, get the word and put it in the map
    ifstream in(file);
    {

      while(getline(in, line)){
        istringstream iss(line);
        while(getline(iss, word, ' ')) {
            words.insert(word);
        }
      }
    }
  }

  ifstream in(gloveFile);
  {
    while(getline(in, line)){
      //for each word that has a gloVe vector
      //get the word
      istringstream iss(line);
      getline(iss, word, ' ');
      vector<float> gloVe;

      //then if the word is in the map, read in all of the floats that make up the vector, add them to gloVe
      //and add the gloVe ctor to the map (couldn't resist, sorry >_>)
      if(words.find(word) != words.end()){

        while(getline(iss, element, ' ')) {
          gloVe.push_back(stof(element));
        }
        word2gl[word] = gloVe;
      }
    }
  }

  return word2gl;
}


vector<Instance> setVector(string hyp1, string hyp2, string ref, unordered_map<string, vector<float>> word2gl) {
  vector<Instance> instances;
  string line;
  string word;

  //hyp1
  {
    //for each word in the input, get the word and add the gloVe ector to the sentence, add sentence to the instance
    ifstream in(hyp1);
    {
      while(getline(in, line)){
        Sentence sentence;
        Instance instance;
        vector<vector<float>> gloVes;
        istringstream iss(line);
        //this will add the semantic expression/vec/whatever
        while(getline(iss, word, ' ')) {
          gloVes.push_back(word2gl[word]);
        }
        sentence.sem = gloVes;
        instance.hyp1 = sentence;
        instances.push_back(instance);
      }
    }
  }


  //hyp2
  {
    //for each word in the input, get the word and add the gloVe ector to the sentence, add sentence to the instance
    ifstream in(hyp2);
    {
      int counter = 0;
      while(getline(in, line)){
        Sentence sentence;
        vector<vector<float>> gloVes;
        istringstream iss(line);
        //this will add the semantic expression/vec/whatever
        while(getline(iss, word, ' ')) {
          gloVes.push_back(word2gl[word]);
        }
        sentence.sem = gloVes;
        instances[counter].hyp2 = sentence;
        counter++;
      }
    }
  }

  //reference
  {
    //for each word in the input, get the word and add the gloVe ector to the sentence, add sentence to the instance
    ifstream in(ref);
    {
      int counter = 0;
      while(getline(in, line)){
        Sentence sentence;
        vector<vector<float>> gloVes;
        istringstream iss(line);
        //this will add the semantic expression/vec/whatever
        while(getline(iss, word, ' ')) {
          gloVes.push_back(word2gl[word]);
        }
        sentence.sem = gloVes;
        instances[counter].ref = sentence;
        counter++;
      }
    }
  }
  return instances;

}

//this sets the bleu score, meteor score, and correct answer
vector<Instance> setBMC(string bleu_file, string meteor_file, string correct, vector<Instance> instances){
  string line;
  string word;

  //BLEU (for french LOSERS) for hyp1-ref, hyp2-ref, hyp1-hyp2 (to be stored in ref)
  {
    //for each word in the input, get the word and add the gloVe ector to the sentence, add sentence to the instance
    ifstream in(bleu_file);
    {
      int counter = 0;
      while(getline(in, line)){
        float BLEU;
        istringstream iss(line);
        //this will add the semantic expression/vec/whatever
        getline(iss, word, '\t');
        BLEU = stof(word);
        instances[counter].hyp1.BLEU = BLEU;

        getline(iss, word, '\t');
        BLEU = stof(word);
        instances[counter].hyp2.BLEU = BLEU;

        getline(iss, word, '\t');
        BLEU = stof(word);
        instances[counter].ref.BLEU = BLEU;

        counter++;
      }
    }
  }

  //Meteor (for AMERICAN WINNERS) for hyp1-ref, hyp2-ref, hyp1-hyp2 (to be stored in ref)
  {
    //for each word in the input, get the word and add the gloVe ector to the sentence, add sentence to the instance
    ifstream in(meteor_file);
    {
      int counter = 0;
      while(getline(in, line)){
        float meteor;
        istringstream iss(line);
        //this will add the semantic expression/vec/whatever
        getline(iss, word, '\t');
        meteor = stof(word);
        instances[counter].hyp1.meteor = meteor;

        getline(iss, word, '\t');
        meteor = stof(word);
        instances[counter].hyp2.meteor = meteor;

        getline(iss, word, '\t');
        meteor = stof(word);
        instances[counter].ref.meteor = meteor;

        counter++;
      }
    }
  }

  //correct answer
  {
    ifstream in(correct);
    {
      int counter = 0;
      while(getline(in, line)){
        vector<vector<float>> gloVes;
        istringstream iss(line);
        //this will add the semantic expression/vec/whatever
        getline(iss, word, '\n');
        instances[counter].correct = stoi(word);
        counter++;
      }
    }
  }
  return instances;
}

vector<Instance> setSyn(string shyp1, string shyp2, string sref, vector<Instance> instances) {
  string line;
  string word;
  string element;

  //hypothesis one sentence vect
  {
    ifstream in(shyp1);
    int counter = 0;
    while(getline(in, line)){
      istringstream iss(line);

      vector<float> sentence_vector;

      while(getline(iss, element, ',')) {
        sentence_vector.push_back(stof(element));
      }
      instances[counter].hyp1.syn = sentence_vector;
      counter++;
    }
  }


  //hypothesis two sentence vect
  {
    ifstream in(shyp2);
    int counter = 0;

    while(getline(in, line)){
      istringstream iss(line);

      vector<float> sentence_vector;

      while(getline(iss, element, ',')) {
        sentence_vector.push_back(stof(element));
      }
      instances[counter].hyp2.syn = sentence_vector;
      counter++;
    }
  }
  
  //ref sentence vect
  {
    ifstream in(sref);
    int counter = 0;

    while(getline(in, line)){
      istringstream iss(line);
      vector<float> sentence_vector;

        while(getline(iss, element, ',')) {
          sentence_vector.push_back(stof(element));
        }
        instances[counter].ref.syn = sentence_vector;
        counter++;
    }
  }
  return instances;
}


Expression buildComputationGraph(Instance instance,
 ComputationGraph& cg, Model m) {
  Expression input_embed = parameter(cg, m.add_parameters({HIDDEN_DIM, INPUT_DIM}));
  Expression W12 = parameter(cg, m.add_parameters({PAIRWISE_DIM, HIDDEN_DIM * 2}));
  Expression b12 = parameter(cg, m.add_parameters({PAIRWISE_DIM}));
  Expression W1r = parameter(cg, m.add_parameters({PAIRWISE_DIM, HIDDEN_DIM * 2}));
  Expression b1r = parameter(cg, m.add_parameters({PAIRWISE_DIM}));
  Expression W2r = parameter(cg, m.add_parameters({PAIRWISE_DIM, HIDDEN_DIM * 2}));
  Expression b2r = parameter(cg, m.add_parameters({PAIRWISE_DIM}));
  Expression V = parameter(cg, m.add_parameters({1, PAIRWISE_DIM * 3 + 4}));
  Expression b = parameter(cg, m.add_parameters({1}));

  // Create embedding from syntax and semantic vector
  Expression hyp1syn = input(cg, {SENTENCE_DIM, 1}, instance.hyp1.syn);
  Expression hyp2syn = input(cg, {SENTENCE_DIM, 1}, instance.hyp2.syn);
  Expression refsyn = input(cg, {SENTENCE_DIM, 1}, instance.ref.syn);

  vector<Expression> hyp1sem_vectors;
  vector<Expression> hyp2sem_vectors;
  vector<Expression> refsem_vectors;
  for (int i = 0; i < instance.hyp1.sem.size(); ++i) {
    hyp1sem_vectors.push_back(input(cg, {GLOVE_DIM, 1}, instance.hyp1.sem[i]));
    hyp2sem_vectors.push_back(input(cg, {GLOVE_DIM, 1}, instance.hyp2.sem[i]));
    refsem_vectors.push_back(input(cg, {GLOVE_DIM, 1}, instance.ref.sem[i])); 
  }
  Expression hyp1sem_matrix = concatenate_cols(hyp1sem_vectors);
  Expression hyp2sem_matrix = concatenate_cols(hyp2sem_vectors);
  Expression refsem_matrix = concatenate_cols(refsem_vectors);

  Expression hyp1sem = reshape(hyp1sem_matrix * transpose(hyp1sem_matrix), {GLOVE_DIM * GLOVE_DIM, 1});
  Expression hyp2sem = reshape(hyp2sem_matrix * transpose(hyp2sem_matrix), {GLOVE_DIM * GLOVE_DIM, 1});
  Expression refsem = reshape(refsem_matrix * transpose(refsem_matrix), {GLOVE_DIM * GLOVE_DIM, 1});

  Expression x1 = input_embed * concatenate({hyp1syn, hyp1sem});
  Expression x2 = input_embed * concatenate({hyp2syn, hyp2sem});
  Expression xref = input_embed * concatenate({refsyn, refsem});

  // Pairwise vectors
  Expression h12 = tanh(W12 * concatenate({x1, x2}) + b12);
  Expression h1r = tanh(W1r * concatenate({x1, xref}) + b1r);
  Expression h2r = tanh(W2r * concatenate({x2, xref}) + b2r);

  // Combination of evaluation input
  Expression BLEU1 = input(cg, instance.hyp1.BLEU);
  Expression BLEU2 = input(cg, instance.hyp2.BLEU);
  Expression meteor1 = input(cg, instance.hyp1.meteor);
  Expression meteor2 = input(cg, instance.hyp2.meteor);
  Expression combined = concatenate({h12, h1r, h2r, BLEU1, BLEU2, meteor1, meteor2});

  Expression y_pred = logistic(V * combined + b);
  Expression y = input(cg, instance.correct);
  Expression loss = binary_log_loss(y_pred, y);

  return y_pred;
}

int main(int argc, char** argv) {
  cnn::Initialize(argc, argv);
  string hyp1 = "../data/hyp1lower.txt";
  string hyp2 = "../data/hyp2lower.txt";
  string ref = "../data/reflower.txt";
  string gloveFile = "../data/glove.6B.50d.txt";
  string bleu = "../data/bleu.txt";
  string meteor = "../data/meteor.txt";
  string correct = "../data/train.gold";
  string shyp1 = "../data/hyp1vectors.txt";
  string shyp2 = "../data/hyp2vectors.txt";
  string sref = "../data/refvectors.txt";

  bool load_model = false;
  string lmodel = "in_model";
  string smodel = "model";

  unordered_map<string, vector<float>> word2gl = getAllWords(hyp1, hyp2, ref, gloveFile);
  vector<Instance> instances = setVector(hyp1, hyp2, ref, word2gl);
  instances = setBMC(bleu, meteor, correct, instances);
  instances = setSyn(shyp1, shyp2, sref, instances);

  vector<unsigned> order(26208);
  for (int i = 0; i < order.size(); ++i) order[i] = i;
  shuffle(order.begin(), order.end(), *rndeng);
  vector<unsigned> training(order.begin(), order.end() - 500);
  vector<unsigned> dev(order.end() - 500, order.end());  

  Model model;

  if (load_model) {
    string fname = lmodel;
    cerr << "Reading parameters from " << fname << "...\n";
    ifstream in(fname);
    assert(in);
    boost::archive::text_iarchive ia(in);
    ia >> model;
  }



}
