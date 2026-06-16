import data.traffic_data as traffic_data
from data.traffic_data import import_records

def main():
    import_records() 
    print(f"Imported {len(traffic_data.data_records)} rows")
    
    if traffic_data:
        print(f"First row: {traffic_data.data_records[1]}")

if __name__ == "__main__":
    main()