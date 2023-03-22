#!/bin/bash
flow table bind --drop -c bind_client1_train_path.json
flow table bind --drop -c bind_client1_val_path.json
flow table bind --drop -c bind_client2_train_path.json
flow table bind --drop -c bind_client2_val_path.json
