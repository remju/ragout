/* Same arithmetic as add()/disguised_add() in TestCase1.c, but renamed and
 * restructured as if it had been copy-pasted into a different file and
 * lightly rewritten - the kind of duplicate a text or AST diff misses. */
int compute_sum(int first, int second)
{
    int result = first + second;
    return result;
}

int is_positive(int value)
{
    if (value > 0)
    {
        return 1;
    }
    return 0;
}
