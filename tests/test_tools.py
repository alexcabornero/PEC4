import sys
import os
import unittest
import pandas as pd
import numpy as np


from src.data.make_dataset import (
    upper_words,
    count_null_values,
    null_values_as_mean,
    merge_df
)

from src.util.tools import (
    load_data,
    get_column_pandas,
    get_column_manual_read,
    get_tracks,
    tracks_with_word,
    count_tracks_decade,
    get_tracks_highest_feature,
    get_feature_stats,
    get_feature_artist,
    get_dist_euclidean,
    get_dist_cosine,
    get_audio_vector,
    get_decades,
    count_artist_tracks_decade,
    get_artists_all_decades,
)


class TestUtils(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Inicialización de la clase """
        #os.chdir('..')
        cls.file_albums = "albums_clean.csv"
        cls.file_artists = "artists_clean.csv"
        cls.file_tracks = "tracks_clean.csv"
        cls.file_data_norm = "data_norm.csv"

        cls.df = load_data(cls.file_data_norm)
        cls.albums = load_data(cls.file_albums)
        cls.artists = load_data(cls.file_artists)
        cls.tracks = load_data(cls.file_tracks)

    def test_make_dataset(self):
        """Test de las funciones del fichero make_dataset.py """
        self.assertIsInstance(self.df, pd.DataFrame)
        self.assertIsInstance(self.albums, pd.DataFrame)
        self.assertIsInstance(self.artists, pd.DataFrame)
        self.assertIsInstance(self.tracks, pd.DataFrame)

        df_test = pd.DataFrame(
            np.array([['patata', 'tomate', 'pepino'],
                      [7, 8, 9],
                      [9, 3, 4.5]]),
            columns=["a", "b", "c"]
        )
        self.assertIsInstance(upper_words(df_test, 'a'), pd.DataFrame)
        self.assertEqual(count_null_values(self.df, 'energy'), 7)
        data = {'a': [8, 9, 10], 'b': [5, 4, 8]}
        df_test = pd.DataFrame(data)
        self.assertIsInstance(null_values_as_mean(df_test, 'a'), pd.DataFrame)
        df1 = pd.DataFrame({
            'Month': ['ene', 'ene', 'feb', 'feb'],
            'Product': ['A', 'B', 'A', 'B'],
            'Sales': [14, 8, 12, 17]
        })
        df2 = pd.DataFrame({
            'Month': ['ene', 'ene', 'feb', 'feb'],
            'Product': ['A', 'B', 'A', 'B'],
            'Sales': [7, 6, 8, 5]
        })
        self.assertIsInstance(
            merge_df(df1, df2, cols=['Product'], suffix=['_suffix', '']),
            pd.DataFrame)

    def test_tools(self):
        """Test de las funciones del fichero tools.py """
        artists_file = os.path.join(
            os.getcwd(), 'data/processed/', self.file_artists)
        r1, t1 = get_column_pandas(artists_file, 'artist_id')
        self.assertIsInstance(r1, list)
        self.assertIsInstance(t1, float)
        r2, t2 = get_column_manual_read(artists_file, 0)
        self.assertIsInstance(r2, list)
        self.assertIsInstance(t2, float)
        self.assertIsInstance(
            get_tracks(self.df, 'popularity', 100), pd.DataFrame)
        words = pd.Series(['agua', 'refresco', 'licor', 'Agua'])
        self.assertEqual(tracks_with_word(words, 'agua'), 2)
        years = np.arange(1980, 1990)
        self.assertEqual(count_tracks_decade(self.df, years), 3903)
        feature = 'popularity_track'
        self.assertIsInstance(
            get_tracks_highest_feature(self.df, feature, years), pd.DataFrame)
        artist = 'Ac/Dc'
        artist_track = get_tracks(self.df, 'name_artist', artist)
        feature = 'popularity'
        mi, me, ma = get_feature_stats(artist_track[feature])
        self.assertEqual(mi, 48)
        self.assertIsInstance(me, float)
        self.assertEqual(ma, 83)
        self.assertIsInstance(
            get_feature_artist(
                self.df, 'name_artist', 'The Beatles'), pd.Series)
        vector1 = [1, 2, 3]
        vector2 = [4, 5, 6]
        self.assertEqual(
            get_dist_euclidean(vector1, vector2), 5.196152422706632)
        self.assertEqual(
            get_dist_cosine(vector1, vector2), 0.9746318461970761)
        features = ['popularity', 'energy']
        self.assertIsInstance(get_audio_vector(self.df, features), list)
        self.assertIsInstance(get_decades(1910, 1940), list)
        years = np.arange(1970, 1980)
        self.assertEqual(
            count_artist_tracks_decade(self.df, 'Ac/Dc', years), 55)
        decadas = get_decades(1920, 1960)
        self.assertIsInstance(
            get_artists_all_decades(self.df, decadas), list)
        self.assertEqual(
            get_artists_all_decades(self.df, decadas), ['Louis Armstrong'])


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestUtils)
    unittest.TextTestRunner(verbosity=2).run(suite)