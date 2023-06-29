import unittest


class MyTestCase(unittest.TestCase):

    def test_image_loading(self):
        from data_loader import DataLoader

        dl = DataLoader(data_set_nr=1,
                        samples_amount=20,
                        shuffle=True,
                        compressed=False)

        X, y = dl.load()
        print('\nX shape: {}\ny shape: {}\n'.format(X.shape, y.shape))
        print('y:{}'.format(y))
        self.assertEqual(X.size, 61440)


class ModelTraining(unittest.TestCase):

    def test_train_svm_on_data(self):
        from models import SVM
        svm = SVM()

        from data_loader import DataLoader
        dl = DataLoader(data_set_nr=1,
                        samples_amount=5000,
                        shuffle=True,
                        compressed=False)
        X, y = dl.load()
        svm.model_train(X, y)
        self.assertTrue(True)


#  TODO If the below test is passed, the project should be done
class ProjectExpectedResult(unittest.TestCase):

    def test_train_and_save_new_model(self):
        import random
        random.seed(10)

        # 1: Unified data loading
        from data_loader import DataLoader
        dl = DataLoader(data_set_nr=1,
                        samples_amount=100,
                        shuffle=True,
                        compressed=False)

        # 2: Automatic model generation
        from transform_model_generation import TransformGen, ModelGen
        transform_sets = TransformGen.get_transform_sets(n_sets=5)
        model_sets = ModelGen.get_classifitcation_models(n_models=5)

        # 3: Combination searching
        # from model_cracker import CombinationCracker
        # cracking_case = CombinationCracker(data=dl, transforms=transform_sets.tf, models=model_sets.mod)


if __name__ == '__main__':
    unittest.main()
