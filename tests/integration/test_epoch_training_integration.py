"""Isolated callback integration, without Transformers, downloads or GPU."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import sys

from training.metadata import EpochMetadataCallback


class EpochIntegration(unittest.TestCase):
    def test_generated_sources_are_synchronized(self):
        from kaggle_training.build_pilot_source import build
        from kaggle_training.build_smoke10_notebook import build as notebook
        root = Path(__file__).resolve().parents[2]
        source = build(root)
        self.assertEqual((root / 'training/kaggle/pilot1000_source.py').read_text(), source)
        scope = {'__name__': 'verification'}
        exec(compile(source, 'pilot', 'exec'), scope)
        self.assertEqual(scope['PAYLOAD']['training/metadata.py'],
                         (root / 'training/metadata.py').read_text())
        generated = json.loads((root / 'temp_session_kaggle/phowhisper2-smoke10/phowhisper2.ipynb').read_text())
        self.assertEqual(generated, notebook(root))
        for cell in generated['cells']:
            compile(''.join(cell['source']), 'cell', 'exec')

    def test_save_before_epoch_and_file_final(self):
        with TemporaryDirectory() as directory:
            args = SimpleNamespace(output_dir=directory)
            state = SimpleNamespace(is_world_process_zero=True, global_step=4, epoch=1.)
            callback = EpochMetadataCallback({})
            callback.on_train_begin(args, state, None)
            callback.on_save(args, state, None)  # Missing directory is not a save.
            checkpoint = Path(directory) / 'checkpoint-4'
            checkpoint.mkdir()
            callback.on_save(args, state, None)
            callback.on_epoch_end(args, state, None)
            callback.on_epoch_end(args, state, None)
            final = Path(directory) / 'weights.pt'
            final.touch()
            callback.final_model(final, state)
            self.assertEqual(len(callback.manifest.data['epochs']), 1)
            self.assertEqual(callback.manifest.data['epochs'][0]['checkpoint_steps'], [4])
            self.assertEqual(callback.manifest.data['final_models'][0]['path'], str(final))

    def test_actual_trainer_wiring(self):
        from kaggle_training.train import run
        with TemporaryDirectory() as directory:
            output = Path(directory)
            manifest = output / 'train.json'
            manifest.write_text('[]')
            args = SimpleNamespace(seed=42, model=directory, revision='local', precision='fp32',
                manifest=manifest, validation=manifest, mode='train', output=output, batch=1,
                accumulation=1, learning_rate=1e-5, max_steps=1, save_steps=10, workers=0,
                require_finite_training=False, resume='old/checkpoint-1')
            callbacks = []
            trainer = Mock()
            trainer.is_world_process_zero.return_value = True
            trainer.state = SimpleNamespace(is_world_process_zero=True, global_step=1, epoch=1.)
            trainer.add_callback.side_effect = callbacks.append
            options = Mock()
            options.output_dir = directory
            options.to_dict.return_value = {'max_steps': 1}
            def train(**kwargs):
                callback = next(c for c in callbacks if isinstance(c, EpochMetadataCallback))
                callback.on_train_begin(options, trainer.state, None)
                callback.on_epoch_end(options, trainer.state, None)
            trainer.train.side_effect = train
            trainer.predict.return_value = SimpleNamespace(metrics={}, predictions=[])
            transformers = Mock()
            transformers.TrainerCallback = type('TrainerCallback', (), {})
            transformers.Seq2SeqTrainingArguments.return_value = options
            transformers.Seq2SeqTrainer.return_value = trainer
            transformers.WhisperProcessor.from_pretrained.return_value.batch_decode.return_value = []
            with patch.dict(sys.modules, transformers=transformers, numpy=Mock(),
                            torch=Mock(__version__='mock'), jiwer=Mock(), huggingface_hub=Mock()):
                run(args)
            trainer.train.assert_called_once_with(resume_from_checkpoint=args.resume)
            data = json.loads((output / 'epochs.json').read_text())
            self.assertEqual(data['invocations'][0]['provenance']['training_arguments'], {'max_steps': 1})
            self.assertEqual(len(data['final_models']), 1)
            self.assertEqual(len(data['epochs']), 1)

    def test_events_and_pruning(self):
        with TemporaryDirectory() as directory:
            args = SimpleNamespace(output_dir=directory)
            state = SimpleNamespace(is_world_process_zero=True, global_step=10, epoch=1.0)
            callback = EpochMetadataCallback({'resume': 'old/checkpoint-5'})
            callback.on_train_begin(args, state, None)
            callback.on_log(args, state, None, logs={'loss': .5, 'grad_norm': 2.0})
            callback.on_epoch_end(args, state, None)
            checkpoint = Path(directory) / 'checkpoint-10'
            checkpoint.mkdir()
            callback.on_save(args, state, None)
            callback.on_evaluate(args, state, None, metrics={'eval_wer': .25})
            checkpoint.rmdir()
            callback.on_train_end(args, state, None)
            callback.final_model(directory, state)
            data = json.loads((Path(directory) / 'epochs.json').read_text())
            self.assertEqual(data['epochs'][0]['metrics'], {'loss': .5, 'grad_norm': 2., 'eval_wer': .25})
            self.assertEqual(data['epochs'][0]['checkpoint_steps'], [10])
            self.assertFalse(data['checkpoints'][0]['exists'])
            self.assertEqual(len(data['final_models']), 1)

    def test_partial_and_nonzero(self):
        with TemporaryDirectory() as directory:
            args = SimpleNamespace(output_dir=directory)
            state = SimpleNamespace(is_world_process_zero=False, global_step=3, epoch=.3)
            callback = EpochMetadataCallback({})
            for event in ('on_train_begin', 'on_log', 'on_evaluate', 'on_epoch_end', 'on_save', 'on_train_end'):
                getattr(callback, event)(args, state, None)
            callback.final_model(directory, state)
            self.assertFalse((Path(directory) / 'epochs.json').exists())
            state.is_world_process_zero = True
            callback.on_train_begin(args, state, None)
            callback.on_epoch_end(args, state, None)
            self.assertEqual(callback.manifest.data['epochs'], [])


if __name__ == '__main__':
    unittest.main()
