"""Pruebas reproducibles, sin Internet, sobre el pipeline y la entrega."""
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import nbformat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.modelado import NUMERICAS, CATEGORICAS, FEATURES, cargar_datos, construir_pipeline, evaluar
from generar_datos import generar


class TestPreentrega10(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X, cls.y, cls.raw = cargar_datos()
        cls.Xtr, cls.Xte, cls.ytr, cls.yte = train_test_split(
            cls.X, cls.y, test_size=.20, random_state=42, stratify=cls.y
        )
        cls.model = construir_pipeline()
        cls.model.fit(cls.Xtr, cls.ytr)
        cls.pred = cls.model.predict(cls.Xte)
        cls.met = evaluar(cls.yte, cls.pred)

    def test_01_dataset_2400_y_21_columnas(self):
        self.assertEqual(self.raw.shape, (2400, 21))

    def test_02_clientes_son_unicos(self):
        self.assertFalse(self.raw.customerID.duplicated().any())

    def test_03_target_binario_y_tiene_ambas_clases(self):
        self.assertEqual(set(self.y.unique()), {0,1})

    def test_04_features_excluyen_id_y_churn(self):
        self.assertEqual(set(self.X.columns), set(FEATURES))
        self.assertFalse('Churn' in self.X.columns or 'customerID' in self.X.columns)

    def test_05_hay_casos_de_nulos_para_imputar(self):
        self.assertGreater(int(self.X.isna().sum().sum()), 0)

    def test_06_division_80_20(self):
        self.assertEqual((len(self.Xtr), len(self.Xte)), (1920,480))
        self.assertTrue(set(self.Xtr.index).isdisjoint(set(self.Xte.index)))

    def test_07_split_estratificado(self):
        self.assertLess(abs(float(self.ytr.mean())-float(self.yte.mean())), .005)

    def test_08_columntransformer_pipelines(self):
        self.assertIsInstance(self.model, Pipeline)
        pre = self.model.named_steps['preprocesador']
        self.assertIsInstance(pre, ColumnTransformer)
        self.assertEqual(set(pre.named_transformers_.keys()), {'numerico','categorico'})
        self.assertIsInstance(pre.named_transformers_['numerico'], Pipeline)

    def test_09_imputer_ajustado_solo_train(self):
        imputer = self.model.named_steps['preprocesador'].named_transformers_['numerico'].named_steps['imputer']
        np.testing.assert_allclose(imputer.statistics_, self.Xtr[NUMERICAS].median().to_numpy())
        self.assertEqual(imputer.n_features_in_, len(NUMERICAS))

    def test_10_encoder_ignora_categorias_nuevas(self):
        encoder = self.model.named_steps['preprocesador'].named_transformers_['categorico'].named_steps['onehot']
        self.assertEqual(encoder.handle_unknown, 'ignore')

    def test_11_predicciones_binarias(self):
        self.assertEqual(len(self.pred), len(self.Xte))
        self.assertTrue(set(self.pred).issubset({0,1}))

    def test_12_metricas_en_rango(self):
        for key in ('accuracy','precision','recall','f1'):
            self.assertGreaterEqual(self.met[key], 0)
            self.assertLessEqual(self.met[key], 1)

    def test_13_confusion_matriz_2x2_y_conserva_filas(self):
        cm = confusion_matrix(self.yte, self.pred, labels=[0,1])
        self.assertEqual(cm.shape, (2,2))
        self.assertEqual(int(cm.sum()), 480)
        self.assertEqual(cm.ravel().tolist(), [self.met[k] for k in ('tn','fp','fn','tp')])

    def test_14_prediccion_nuevo_cliente_nulos_y_categorias_desconocidas(self):
        nuevo = self.Xte.iloc[[0]].copy()
        nuevo["tenure"] = nuevo["tenure"].astype(float)
        nuevo.loc[:, 'tenure'] = np.nan
        nuevo.loc[:, 'Contract'] = 'Contrato desconocido'
        self.assertIn(int(self.model.predict(nuevo)[0]), {0,1})

    def test_15_dataset_generado_reproducible(self):
        a = generar(12, 10)
        b = generar(12, 10)
        pd.testing.assert_frame_equal(a,b)

    def test_16_evidencias_existen_y_totales(self):
        m = json.loads((ROOT/'evidencia/metricas.json').read_text(encoding='utf8'))
        self.assertEqual(m['filas_totales'], 2400)
        self.assertEqual(m['filas_train'], 1920)
        self.assertEqual(m['filas_test'], 480)
        self.assertEqual(m['tp'] + m['tn'] + m['fp'] + m['fn'], 480)
        self.assertTrue((ROOT/'evidencia/matriz_confusion.png').exists())
        self.assertTrue((ROOT/'evidencia/reporte_clasificacion.txt').exists())
        self.assertAlmostEqual(m['f1'], self.met['f1'], places=6)

    def test_17_notebook_esta_ejecutado_y_con_matriz(self):
        nb = nbformat.read(ROOT/'notebooks/modelado_metricas.ipynb', as_version=4)
        self.assertGreaterEqual(sum(c.cell_type=='markdown' for c in nb.cells),6)
        code_cells = [c for c in nb.cells if c.cell_type=='code']
        self.assertTrue(all(c.execution_count is not None for c in code_cells))
        self.assertFalse(any(o.output_type=='error' for c in code_cells for o in c.outputs))
        self.assertTrue(any(o.data.get('image/png') for c in code_cells for o in c.outputs if o.output_type in {'display_data','execute_result'}))

    def test_18_conclusion_es_markdown_y_menciona_falsos_negativos(self):
        nb = nbformat.read(ROOT/'notebooks/modelado_metricas.ipynb', as_version=4)
        all_md = ' '.join(c.source.lower() for c in nb.cells if c.cell_type=='markdown')
        self.assertIn('falso negativo', all_md)
        self.assertIn('recall', all_md)
        self.assertIn('impacto', all_md) if 'impacto' in all_md else self.assertIn('negocio', all_md)

    def test_19_readme_estructura_e_instrucciones(self):
        r=(ROOT/'README.md').read_text(encoding='utf8').lower()
        for v in ('requisitos', 'ejecuci', 'notebooks/modelado_metricas.ipynb', 'recall', 'ibm', 'simulado'):
            self.assertIn(v,r)

    def test_20_gitignore_no_excluye_notebook(self):
        ignore=(ROOT/'.gitignore').read_text(encoding='utf8')
        self.assertIn('__pycache__/',ignore)
        self.assertIn('.ipynb_checkpoints/', ignore)
        self.assertNotIn('*.ipynb', ignore)


if __name__=='__main__':
    unittest.main()
