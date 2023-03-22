#!/bin/bash
flow table bind --drop -c bind_guest_train_path.json
flow table bind --drop -c bind_guest_valid_path.json
flow table bind --drop -c bind_guest_test_path.json
flow table bind --drop -c bind_host_train_path.json
flow table bind --drop -c bind_host_valid_path.json
flow table bind --drop -c bind_host_test_path.json