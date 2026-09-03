/* Same bubble-sort algorithm as bubble_sort() in TestCase3.c, renamed and
 * lightly restructured (swap variable declared inline, loop variables
 * renamed) - a bigger, more realistic copy-paste case than a one-liner. */
void sort_ascending(int *values, int count)
{
    int outer, inner;

    for (outer = 0; outer < count - 1; outer++)
    {
        for (inner = 0; inner < count - outer - 1; inner++)
        {
            if (values[inner] > values[inner + 1])
            {
                int swap_tmp = values[inner];
                values[inner] = values[inner + 1];
                values[inner + 1] = swap_tmp;
            }
        }
    }
}

/* Unrelated bigger function - should NOT be flagged as a duplicate of the
 * sort functions above. */
int binary_search(const int *sorted, int n, int target)
{
    int low = 0;
    int high = n - 1;

    while (low <= high)
    {
        int mid = low + (high - low) / 2;

        if (sorted[mid] == target)
        {
            return mid;
        }
        else if (sorted[mid] < target)
        {
            low = mid + 1;
        }
        else
        {
            high = mid - 1;
        }
    }

    return -1;
}
