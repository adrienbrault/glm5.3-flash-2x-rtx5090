#!/usr/bin/env python3
"""Operator-only: clone a stopped served container's launch config with an A/B env.
Preserves command, mounts, GPU requests, ports, shmem, limits, and named networks.
Does not delete or modify the source container. Refuses an unacknowledged stop.
"""
import argparse, http.client, json, socket, subprocess
ap=argparse.ArgumentParser();ap.add_argument('--source',required=True)
ap.add_argument('--name',required=True);ap.add_argument('--image',required=True)
ap.add_argument('--env-file',required=True);ap.add_argument('--stop-source',action='store_true');a=ap.parse_args()
r=json.loads(subprocess.check_output(['docker','inspect',a.source]))[0]
if r['State']['Running']:
    if not a.stop_source: raise SystemExit('Source is running; add --stop-source for the authorized A/B interruption.')
    subprocess.run(['docker','stop',a.source],check=True)
config=r['Config']; config['Image']=a.image;config['Hostname']=a.name
settings={}
for line in open(a.env_file):
    line=line.strip()
    if line and not line.startswith('#'):
        k,v=line.split('=',1);settings[k]=v
existing=dict(item.split('=',1) for item in config.get('Env',[]))
# Prevent inherited stats files, instrumentation, or exchange-mode knobs contaminating either arm.
for k in list(existing):
    if (k.startswith('EXL3_MOE_CPU_SWAP') or k.startswith('EXL3_MOE_CPU_SCORE')) or k in ('EXL3_MOE_CPU_SPLIT_STATS','EXL3_MOE_CPU_PROF','EXL3_MOE_CPU_SPLIT_PROF'):
        existing.pop(k)
existing.update(settings);config['Env']=[k+'='+v for k,v in existing.items()]
# API accepts the image's Config plus HostConfig. Named volume data comes from HostConfig.
config.pop('OnBuild',None);config.pop('ArgsEscaped',None);config['HostConfig']=r['HostConfig']
config['HostConfig'].pop('ContainerIDFile',None)
# Keep anonymous/data volumes too: Config.Volumes alone would create fresh empty ones.
hc=config['HostConfig']; represented={m['Target'] for m in hc.get('Mounts',[]) or []}
represented.update(b.split(':')[1] for b in hc.get('Binds',[]) or [] if ':' in b)
extra=list(hc.get('Mounts',[]) or [])
for mount in r.get('Mounts',[]):
    if mount['Destination'] in represented or mount['Type'] not in ('bind','volume'): continue
    extra.append(dict(Type=mount['Type'],Source=mount.get('Name',mount['Source']) if mount['Type']=='volume' else mount['Source'],
                      Target=mount['Destination'],ReadOnly=not mount['RW']))
if extra: hc['Mounts']=extra
networks={}
for name in r['NetworkSettings'].get('Networks',{}):
    if name not in ('host','none','bridge'): networks[name]={}
if networks: config['NetworkingConfig']={'EndpointsConfig':networks}
class Docker(http.client.HTTPConnection):
    def connect(self):
        self.sock=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);self.sock.connect('/var/run/docker.sock')
c=Docker('localhost');version=json.loads(subprocess.check_output(['docker','version','--format','{{json .Server}}']))['ApiVersion']
from urllib.parse import quote
c.request('POST',f'/v{version}/containers/create?name={quote(a.name)}',json.dumps(config),{'Content-Type':'application/json'})
response=c.getresponse();data=response.read()
if response.status!=201: raise SystemExit(f'create failed ({response.status}): {data.decode()}')
identifier=json.loads(data)['Id'];subprocess.run(['docker','start',identifier],check=True)
print('Started',a.name,'with preserved served configuration. Source remains stopped; restore with docker start',a.source)
