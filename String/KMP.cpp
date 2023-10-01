#include <bits/stdc++.h>

using namespace std;

const int N = 1e5 + 5, M = 1e6 + 5;

int n, m, ne[N];
char P[N], S[M]; // P是模版串，S是大串

void KMP()
{
    cin >> n >> P + 1 >> m >> S + 1;

    for (int i = 2, j = 0; i <= n; i ++ )
    {
        while (j && P[j + 1] != P[i]) j = ne[j];
        if (P[i] == P[j + 1]) j ++;
        ne[i] = j;
    }

    for (int i = 1, j = 0; i <= m; i ++ )
    {
        while (j && P[j + 1] != S[i]) j = ne[j];
        if (P[j + 1] == S[i]) j ++;
        if (j == n)
        {
            cout << i - n << " ";
            j = ne[j];
        }
    }
}
