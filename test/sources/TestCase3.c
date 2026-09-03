#include <stdio.h>

/* Bigger duplicate case: a full bubble sort, ~20 lines. Compare against
 * sort_ascending() in TestCase4.c, which does the same thing with renamed
 * variables and a slightly restructured swap step. */
void bubble_sort(int *arr, int n)
{
    int i, j, temp;

    for (i = 0; i < n - 1; i++)
    {
        for (j = 0; j < n - i - 1; j++)
        {
            if (arr[j] > arr[j + 1])
            {
                temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }
}

void print_array(int *arr, int n)
{
    int i;

    for (i = 0; i < n; i++)
    {
        printf("%d ", arr[i]);
    }
    printf("\n");
}
