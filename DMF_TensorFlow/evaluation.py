import math
import pandas as pd
from progressbar import ProgressBar


def evaluate(model, sess, evaluation_data, top_k):
    metrics = Metrics(top_k)

    pos_users, pos_items = evaluation_data[0], evaluation_data[1]
    neg_users, neg_items = evaluation_data[2], evaluation_data[3]

    pos_scores = list(model.predict(sess, {model.user: pos_users, model.item: pos_items}))
    neg_scores = []
    start = 0
    while start < len(neg_items):
        neg_scores.extend(list(model.predict(sess, {model.user: neg_users[start:start+1024], model.item: neg_items[start:start+1024]})))
        start += 1024

    metrics.subjects = [pos_users, pos_items, pos_scores, neg_users, neg_items, neg_scores]

    hit_ratio = metrics.compute_hit_ratio()
    ndcg = metrics.compute_ndcg()
    return hit_ratio, ndcg


class Metrics():

    def __init__(self, top_k):
        self._top_k = top_k
        self._subjects = None

    @property
    def subjects(self):
        return self._subjects

    @subjects.setter
    def subjects(self, subjects):
        pos_users, pos_items, pos_scores = subjects[0], subjects[1], subjects[2]
        neg_users, neg_items, neg_scores = subjects[3], subjects[4], subjects[5]
        pos_df = pd.DataFrame({
            'user': pos_users,
            'pos_item': pos_items,
            'pos_score': pos_scores})
        full_df = pd.DataFrame({
            'user': neg_users + pos_users,
            'item': neg_items + pos_items,
            'score': neg_scores + pos_scores})
        full_df = pd.merge(full_df, pos_df, on=['user'], how='left')
        full_df['rank'] = full_df.groupby('user')['score'].rank(method='first', ascending=False)
        full_df.sort_values(['user', 'rank'], inplace=True)
        self._subjects = full_df

    def compute_hit_ratio(self):
        top_k = self._subjects[self._subjects['rank']<=self._top_k]
        pos_in_top_k = top_k[top_k['pos_item'] == top_k['item']]
        return len(pos_in_top_k) / self._subjects['user'].nunique()

    def compute_ndcg(self):
        top_k = self._subjects[self._subjects['rank']<=self._top_k]
        pos_in_top_k = top_k[top_k['pos_item'] == top_k['item']].copy()
        pos_in_top_k['ndcg'] = pos_in_top_k['rank'].apply(lambda x: math.log(2) / math.log(1 + x))
        return pos_in_top_k['ndcg'].sum() / self._subjects['user'].nunique()
