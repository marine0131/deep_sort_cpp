#include <iostream>
#include <vector>
#include "nms.hpp"
using namespace std;

int main(int argc, char** argv)
{
    vector<vector<float> > boxes(2, vector<float>(4, 0));
    boxes[0][0] = 1;
    boxes[0][1] = 1;
    boxes[0][2] = 2;
    boxes[0][3] = 2;
    boxes[1][0] = 1.1;
    boxes[1][1] = 1.1;
    boxes[1][2] = 2.2;
    boxes[1][3] = 2.2;

    vector<float> scores = {0.9, 1.7};
    vector<int> pick = nms(boxes, 0.5, scores);
    for(vector<int>::iterator i=pick.begin(); i!=pick.end(); ++i)
        cout << *i << endl;
}
