"""Offline CPU export; no ROS or robot interfaces. Never overwrite an existing bundle."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import torch
from torch import nn
import onnxruntime as ort
import yaml

class Loader(yaml.SafeLoader):
    pass
Loader.add_multi_constructor('tag:yaml.org,2002:python/', lambda loader, tag, node:
    loader.construct_sequence(node) if isinstance(node, yaml.SequenceNode) else
    loader.construct_mapping(node) if isinstance(node, yaml.MappingNode) else loader.construct_scalar(node))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--env', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args=p.parse_args()
    cfg=yaml.load(args.env.read_text(), Loader=Loader)
    profile=cfg.get('deployment_observation_profile','unified_external_xy_v1')
    expected='proprioceptive_flat:proprio_features' if profile=='proprioceptive_flat_v1' else 'unified_asymmetric:actor_features'
    assert cfg['observations']['policy']['unified_features']['func'].endswith(expected)
    assert abs(cfg['sim']['dt']*cfg['decimation']-.02)<1e-9
    ckpt=torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    assert not any('normalizer' in k for k in ckpt), 'inspect observation normalization first'
    if profile=='proprioceptive_flat_v1':
        assert ckpt.get('iter',0)>13493, 'Do not relabel the unadapted 13493 checkpoint as proprioceptive'
    state=ckpt['model_state_dict']
    assert state['actor.0.weight'].shape==(512,483)
    assert state['critic.0.weight'].shape==(512,560)
    actor=nn.Sequential(nn.Linear(483,512),nn.ELU(),nn.Linear(512,256),nn.ELU(),
                        nn.Linear(256,128),nn.ELU(),nn.Linear(128,12)).eval()
    actor.load_state_dict({k[6:]:v for k,v in state.items() if k.startswith('actor.')}, strict=True)
    args.output.mkdir(parents=True, exist_ok=False)
    target=args.output/'policy.onnx'
    torch.onnx.export(actor, torch.zeros(1,483), str(target), input_names=['obs'], output_names=['actions'],
                      dynamic_axes={'obs':{0:'batch'}, 'actions':{0:'batch'}}, opset_version=17, dynamo=False)
    session=ort.InferenceSession(str(target),providers=['CPUExecutionProvider'])
    rng=np.random.default_rng(20261023)
    maximum=0.
    for batch in (1,8,32):
        x=rng.normal(size=(batch,483)).astype(np.float32)
        with torch.inference_mode(): expected=actor(torch.from_numpy(x)).numpy()
        actual=session.run(None,{'obs':x})[0]
        np.testing.assert_allclose(actual,expected,rtol=2e-5,atol=2e-5)
        maximum=max(maximum,float(np.max(np.abs(actual-expected))))
    # Save data-only YAML, no Python object constructors on deployment host.
    (args.output/'env.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    pattern=cfg['scene']['height_scanner']['pattern_cfg']
    assert pattern['ordering']=='xy' and pattern['size']==[1.6,1.0] and pattern['resolution']==.1
    # torch.arange/meshgrid matches Isaac grid_pattern including float32 rounding.
    x=torch.arange(-.8,.8+1e-9,.1); y=torch.arange(-.5,.5+1e-9,.1)
    xx,yy=torch.meshgrid(x,y,indexing='xy')
    grid=torch.stack([xx.flatten(),yy.flatten()],dim=-1).numpy()
    assert grid.shape==(187,2)
    np.savez(args.output/'height_scan_contract.npz',grid_xy=grid)
    scan={'obs_dim':483,'height_scan_dim':187,'observation_slices':{'height_scan':[66,253]},
          'height_scan':{'grid_xy_source':'height_scan_contract.npz:grid_xy','clip':[-1,1],
                         'scale':1.,'offset':.5,'frame':'base_yaw_aligned'}}
    (args.output/'height_scan_contract.yaml').write_text(yaml.safe_dump(scan,sort_keys=False))
    sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    manifest={'schema_version':1,'name':f'Go2-X5 unified483 iteration {ckpt.get("iter")}',
              'source_checkpoint':args.checkpoint.name,'source_sha256':sha(args.checkpoint),
              'observation_profile':profile,'iteration':ckpt.get('iter'),'actor_dim':483,'critic_dim':560,'actions':12,
              'artifacts':{name:{'path':file,'sha256':sha(args.output/file)} for name,file in
                           [('policy','policy.onnx'),('env','env.yaml'),('height_contract','height_scan_contract.yaml'),('height_grid','height_scan_contract.npz')]}}
    (args.output/'policy_bundle.json').write_text(json.dumps(manifest,indent=2)+'\n')
    report={'torch':torch.__version__,'onnxruntime':ort.__version__,'batches':[1,8,32],
            'max_absolute_error':maximum,'scope':'CPU synthetic inference equivalence, not hardware acceptance'}
    (args.output/'export_check.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
