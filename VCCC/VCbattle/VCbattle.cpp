#include "stdafx.h"
#include <iostream>
#include <boost/functional/hash.hpp>

using namespace std;
using namespace boost;

class X {
public:
	X(int i) : val_(i) {}
	int get_val() const { return val_; }
private:
	int val_;
};

//construct a hash function for class X
size_t hash_value(const X &x)
{
	boost::hash<int> hasher;
	return hasher(x.get_val());
}

int main(int argc, char *argv[])
{
	boost::hash<std::string> string_hash;
	boost::hash<double> double_hash;
	//boost::hash<vector<int> > vector_hash;
	//hash for user defined type
	boost::hash<X> customerX_hash;

	//vector()
	X x(32);

	size_t h = string_hash("hash me");
	cout << h << "/n";

	h = double_hash(3.1415927);
	cout << h << "/n";

	//h = vector_hash(v);
	/*cout << h << "/n";

	v.push_back(10);
	h = vector_hash(v);
	cout << h << "/n";*/

	h = customerX_hash(x);
	cout << h << "/n";

	system("PAUSE");
	return 0;
}