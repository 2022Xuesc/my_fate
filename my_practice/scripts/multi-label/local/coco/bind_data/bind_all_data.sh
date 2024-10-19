#!/bin/bash
flow table bind --drop -c bind_guest_train_path.json
flow table bind --drop -c bind_guest_val_path.json
flow table bind --drop -c bind_host_train_path.json
flow table bind --drop -c bind_host_val_path.json
