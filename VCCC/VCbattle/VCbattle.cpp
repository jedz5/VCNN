#include "stdafx.h"
#include<iostream>
#include<pybind11/pybind11.h>
#include<pybind11/numpy.h>
#include <pybind11/stl.h>
namespace py = pybind11;
using namespace std;
const int  bFieldWidth = 17;
const int  bFieldHeight = 11;
class bhex {
public:
	int x = -1, y = -1;
	bhex() :x(0), y(0) {  }; //cout << "construct invoke!" << endl;
	bhex(const bhex& bh) {
		this->x = bh.x;
		this->y = bh.y;
		//printf("copy(%d,%d) construct invoke!\n", bh.x, bh.y);
	}
	bhex(int xx, int yy) :x(xx), y(yy) {};
	bhex& operator= (const bhex& bh);
	bool operator== (const bhex& bh);
	std::vector<bhex> get_neighbor();
	bhex get_clone();
	int to_position_id();
};
bool check_position(int x, int y){
    return (x >= 1 && x < (bFieldWidth - 1) && (y >= 0 && y < bFieldHeight));
}
void check_and_push(int x, int y, std::vector<bhex>& adj) {
	if (check_position(x,y))
		adj.push_back(bhex(x, y));
}
int get_distance_inner(int sx,int sy, int dx, int dy) {
	int sx1 = int(sx + sy*0.5);
	int dx1 = int(dx + dy*0.5);
	int xDst = dx1 - sx1;
	int yDst = dy - sy;
	if ((xDst >= 0 && yDst >= 0) || (xDst < 0 && yDst < 0))
		return std::max(abs(yDst), abs(xDst));
	return abs(yDst) + abs(xDst);
}
int get_distance(const bhex& self, const bhex& dest) {
	return get_distance_inner(self.x,self.y,dest.x,dest.y);
}
//int get_distance(const py::object& self, const py::object& dest) {
//	return get_distance(self.attr("x").cast<int>(), self.attr("y").cast<int>(), dest.attr("x").cast<int>(), dest.attr("y").cast<int>());
//}
bhex& bhex::operator= (const bhex& bh) {
	//cout << "copy!" << endl;
	this->x = bh.x;
	this->y = bh.y;
	return *this;
}
bool bhex::operator== (const bhex& bh) {
	return this->x == bh.x && this->y == bh.y;
}

std::vector<bhex> bhex::get_neighbor() {
	int zigzag = 0;
	if (this->y % 2 != 0)
		zigzag = 1;
	std::vector<bhex> adj;
	adj.reserve(6);
	check_and_push(this->x - 1, this->y, adj);
	check_and_push(this->x + 1, this->y, adj);
	check_and_push(this->x + 1 - zigzag, this->y - 1, adj);
	check_and_push(this->x - zigzag, this->y - 1, adj);
	check_and_push(this->x + 1 - zigzag, this->y + 1, adj);
	check_and_push(this->x - zigzag, this->y + 1, adj);
	return adj;
}
bhex bhex::get_clone()
{
	return bhex(this->x,this->y);
}
int bhex::to_position_id()
{
	return this->y * bFieldWidth + this->x;
}
enum action_query_type {
	can_move = 0,
	move_to,
	can_attack,
	attack_target,
	attack_from,
	spell
};
class bstack :public bhex{
public:
	int amount = 0;
	//int attack = 0;
	//int defense = 0;
	//int max_damage = 0;
	//int min_damage = 0;
	//int first_HP_Left = 0;
	//int health = 0;
	int side = 0;
	//bool had_moved = false;
	//bool had_retaliated = false;
	//bool had_waited = false;
	//bool had_defended = false;
	int speed = 0;
	//int luck = 0;
	//int morale = 0;
	//int id = 0;
	int shots = 10;
		// hex_type = hexType.creature
		//
	//int by_AI = 1;
	//std::string name = "unKnown";
	//int slotId = 0;
	//bool is_wide = false;
	bool is_fly = false;
	bool is_shooter = false;
	//bool block_retaliate = false;
	//bool attack_nearby_all = false;
	//bool wide_breath = false;
	//bool infinite_retaliate = false;
	//bool attack_twice = false;
	//int amount_base = 0;
	//in_battle = 0;  //Battle()
	bool equals(const bhex& bh);
	//bool equals(const bstack& bh);
	bool is_alive() const;
	bool can_shoot(const std::vector<bstack>& stacks);
};

bool bstack::equals(const bhex& bh)
{
	return this->x = bh.x && this->y == bh.y;
}

//bool bstack::equals(const bstack& bs)
//{
//	return this->x = bs.x && this->y == bs.y && this->id == bs.id;
//}

bool bstack::is_alive() const {
	return this->amount > 0;
}
bool bstack::can_shoot(const std::vector<bstack>& stacks) {
	if (!this->is_shooter || this->shots <= 0)
		return false;
	for (const bstack& enemy : stacks) {
		if (enemy.side != this->side && enemy.is_alive() && get_distance(*this, enemy) == 1)
			return false;
	}
	return true;
}
py::object get_global_state(py::object& in_self,std::vector<py::object>& in_stacks,int query_type = -1,bool exclude_me = true) {
	bstack self;
	std::vector<bstack> stacks;
	stacks.reserve(14);
	self.x = in_self.attr("x").cast<int>();
	self.y = in_self.attr("y").cast<int>();
	self.side = in_self.attr("side").cast<int>();
	self.amount = in_self.attr("amount").cast<int>();
	self.speed = in_self.attr("speed").cast<int>();
	self.is_fly = in_self.attr("is_fly").cast<bool>();
	self.is_shooter = in_self.attr("is_shooter").cast<bool>();
	self.shots = in_self.attr("shots").cast<int>();

	for (auto st : in_stacks) {
		bstack bst;
		bst.x = st.attr("x").cast<int>();
		bst.y = st.attr("y").cast<int>();
		bst.side = st.attr("side").cast<int>();
		bst.amount = st.attr("amount").cast<int>();
		bst.speed = st.attr("speed").cast<int>();
		bst.is_fly = st.attr("is_fly").cast<bool>();
		bst.is_shooter = st.attr("is_shooter").cast<bool>();
		bst.shots = st.attr("shots").cast<int>();
		stacks.push_back(bst);
	}
	//std::cout << "here!!" << std::endl;
	py::array_t<int, py::array::c_style> a({ 11, 17});
	auto bf = a.mutable_unchecked<2>();
	for (ssize_t i = 0; i < 11; i++)
		for (ssize_t j = 0; j < 17; j++)
			if (j == 0 || j == 16)
				bf(i, j) = 100;
			else
				bf(i, j) = -1;
	for (auto st : stacks) {
		if (st.is_alive())
			if (st.side == self.side)
				bf(st.y,st.x) = 400;
			else
				bf(st.y, st.x) = 200;
	}
	std::vector<bhex> travellers;
	travellers.reserve(30);
	bf(self.y, self.x) = self.speed;
	travellers.push_back(self);
	if (!self.is_fly) {
		while (travellers.size() > 0)
		{
			bhex& current = travellers.back();
			travellers.pop_back();
			int speed_left = bf(current.y, current.x) - 1;
			for (bhex& adj : current.get_neighbor()) {
				if (bf(adj.y, adj.x) < speed_left) {
					bf(adj.y, adj.x) = speed_left;
					if (speed_left > 0) {
						travellers.push_back(adj);
						if (query_type == action_query_type::can_move) {
							return py::bool_(true);
						}
					}
				}
			}
		}
	}
	else { // fly
		for (int ii = 0; ii < bFieldHeight; ii++)
			for (int jj = 0; jj < bFieldWidth - 1; jj++) {
				if (bf(ii, jj) > 50)
					continue;
				int d = get_distance(self, bhex(jj, ii));
				if (d > 0 && d <= self.speed) {
					bf(ii, jj) = self.speed - d;
					if (query_type == action_query_type::can_move) {
						return py::bool_(true);
					}
				}
			}
	}
	// no space to move
	if (query_type == action_query_type::can_move) {
		return py::bool_(false);
	}
	//accessable  end
	//attackable begin
	bool can_shoot = self.can_shoot(stacks);
	for (auto st : stacks) {
		if (st.amount <= 0)
			continue;
		if (st.side != self.side) {
			if (can_shoot) {
				bf(st.y, st.x) = 201;
				if(query_type == action_query_type::can_attack)
					return py::bool_(true);
			}
			else {
				for(auto neib : st.get_neighbor())
					if (bf(neib.y, neib.x) >= 0 && bf(neib.y, neib.x) < 50) {
						bf(st.y, st.x) = 201;
						if (query_type == action_query_type::can_attack)
							return py::bool_(true);
						break;
					}
			} 
		}
	}
	// no target to reach
	if (query_type == action_query_type::can_attack)
		return py::bool_(false);
	if(exclude_me)
		bf(self.y, self.x) = 401;
	return a;
}

PYBIND11_MODULE(VCbattle, m) {
	m.def("get_global_state", &get_global_state, py::arg("self"), py::arg("stacks"),py::arg("query_type") = -1, py::arg("exclude_me") = true, py::return_value_policy::move); //
	m.def("get_distance", &get_distance);
	m.def("check_position", &check_position);
	
	py::class_<bhex>(m, "BHex")
		.def(py::init())
		.def(py::init<int, int>())
		.def_readwrite("x",&bhex::x)
		.def_readwrite("y", &bhex::y)
		.def("__eq__", &bhex::operator==)
		.def("get_neighbor", &bhex::get_neighbor, py::return_value_policy::move)
		.def("__copy__", &bhex::get_clone, py::return_value_policy::move)
		.def("to_position_id", &bhex::to_position_id);
}

//vector<bhex> get_n2() {
//	vector<bhex> nb;
//	nb.reserve(6);
//	for (int i = 0; i < 6; i++) {
//		nb.push_back(bhex(i, i));
//		cout << "***" << endl;
//	}
//	cout << "------------------------------"<<endl;
//	return nb;
//}
//int main() {
//	//unique_ptr < vector<unique_ptr<bhex>>> xx = get_n();
//	vector<bhex> xx = get_n2();
//	auto yy = xx;
//	yy[0].x = 111;
//	for (int i = 0; i < 6; i++) {
//		cout << xx[i].x<< endl;
//	}
//}