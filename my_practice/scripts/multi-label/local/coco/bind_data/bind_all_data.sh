#!/bin/bash
flow table bind --drop -c bind_guest_train_path.json
flow table bind --drop -c bind_guest_val_path.json
flow table bind --drop -c bind_host1_train_path.json
flow table bind --drop -c bind_host1_val_path.json

flow table bind --drop -c bind_host2_train_path.json
flow table bind --drop -c bind_host2_val_path.json