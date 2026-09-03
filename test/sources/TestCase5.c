/* Caller/callee pair: process_order() calls validate_total(). These are NOT
 * duplicates, but they share vocabulary and the callee's name, which is enough
 * to pull them together in embedding space. Guarding against this is what
 * min_structure and filter_callers are for. */
int validate_total(int total)
{
    if (total < 0)
    {
        return 0;
    }
    return 1;
}

int process_order(int total, int quantity)
{
    int subtotal = total * quantity;

    if (!validate_total(subtotal))
    {
        return -1;
    }

    return subtotal;
}
