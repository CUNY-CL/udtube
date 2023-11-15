import argparse
from ast import arg
import sqlite3



def set_up_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_db')
    parser.add_argument('table_name')
    return parser

if __name__ == "__main__":
    parser = set_up_parser()
    args = parser.parse_args()
    word = 'αγγειίτιδα'
    with sqlite3.connect(args.input_db) as connection:
        cursor = connection.cursor()
        query = f"""
        WITH totals AS (
            SELECT SUM(count) AS total_count
            FROM {args.table_name}
        ),
        all_freqs AS (
            SELECT
            *,
            (CAST(count AS FLOAT) / totals.total_count) AS abs_freq
            FROM {args.table_name}, totals
        ),
        lemma_freqs AS (
            SELECT
            *,
            (CAST(SUM(count) AS FLOAT) / totals.total_count) AS lem_freq
            FROM {args.table_name}, totals
            GROUP BY lemma
        ),
        feats_freqs AS (
            SELECT
            *,
            (CAST(SUM(count) AS FLOAT) / totals.total_count) AS feats_freq
            FROM {args.table_name}, totals
            GROUP BY feats
        ),
        final_freqs AS (
            af.*,
            lf.lem_freq, 
            ff.feats_freq
            FROM all_freqs af
            LEFT JOIN lemma_freqs lf ON af.lemma = lf.lemma
            LEFT JOIN feats_freqs ff ON af.feats = ff.feats
        )
        SELECT 
        *,
        abs_freq - (lem_freq*feats_freq) as absolute_divergence,
        log2(lem_freq*feats_freq) - log2(lem_freq) - log2(feats_freq) as log_odds_ratio,
        (abs_freq - (lem_freq * feats_freq)) / (SQRT((VAR_SAMP(lem_freq)*VAR_SAMP(feats_freq)) + (VAR_SAMP(lem_freq)*feats_freq) + (lem_freq*VAR_SAMP(feats_freq))) as std_score
        FROM
        final_freqs
        """
        res = cursor.execute(query)
        i = 0
        for r in res.fetchall():
            i += 1
            print(r)
            if i == 10:
                break