#include <bits/stdc++.h>
using namespace std;

bool check(int mid)
{
	return true;
	//return false;
}

void solve()
{
	//板子1
    int x, y;
    int l = x, r = y;
    while (l < r)//求区间[x, y]上满足性质的最小值
    {
        int mid = l + r >> 1;
        if (check(mid)) r = mid;
        else l = mid + 1;
    }
    cout << l; //此时l == r

	//板子2
	int l = x, r = y;
    while (l < r)//求区间[x, y]满足性质的最大值
    {
        int mid = l + r + 1 >> 1;
        if (check(mid)) l = mid;
        else r = mid - 1;
    }
	cout << l; //此时l == r
}