import dask.dataframe as dd
import pandas as pd
from tqdm import tqdm
from dask.diagnostics import ProgressBar
from tenacity import retry, stop_after_attempt, wait_exponential
import requests
import os

class DataFrameProcessor:

    def __init__(self, input_file, output_file, storage_file, api_url, npartitions=4):
        """
        Initialize the processor.
        
        Args:
            input_file (str): Path to the input CSV file.
            output_file (str): Path to save the processed output.
            storage_file (str): Path for persistent storage.
            api_url (str): API URL to call with the input values.
            npartitions (int): Number of partitions for Dask DataFrame.
        """
        self.input_file = input_file
        self.output_file = output_file
        self.storage_file = storage_file
        self.api_url = api_url
        self.npartitions = npartitions
        self.storage = self._load_storage()

    def _load_storage(self):
        """Load persistent storage or create an empty DataFrame."""
        if os.path.exists(self.storage_file):
            return pd.read_csv(self.storage_file).set_index("input")
        else:
            return pd.DataFrame(columns=["input", "result"]).set_index("input")

    def _save_storage(self):
        """Save the persistent storage to a CSV file."""
        self.storage.to_csv(self.storage_file)

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=1, max=10))
    def call_api(self, value):
        """Call the external API with retries."""
        response = requests.get(f"{self.api_url}?param={value}")
        response.raise_for_status()
        return response.json().get("result")

    def process_row(self, row):
        """Process a single row, checking storage and calling the API if needed."""
        value = row["input_column"]
        if value in self.storage.index and pd.notnull(self.storage.loc[value, "result"]):
            return self.storage.loc[value, "result"]
        else:
            try:
                result = self.call_api(value)
                self.storage.loc[value] = result
                return result
            except Exception:
                return None  # Handle API failures gracefully

    def _process_chunk(self, chunk):
        """Process a Dask DataFrame chunk with a progress bar."""
        results = []
        for _, row in tqdm(chunk.iterrows(), total=len(chunk), desc="Processing chunk"):
            results.append(self.process_row(row))
        chunk["result_column"] = results
        return chunk

    def _load_input_data(self):
        """Load input data and find rows to process."""
        if os.path.exists(self.output_file):
            output_df = pd.read_csv(self.output_file)
            input_df = pd.read_csv(self.input_file)
            merged_df = input_df.merge(output_df, on="input_column", how="left", suffixes=("", "_processed"))
            to_process = merged_df[merged_df["result_column"].isnull()]
        else:
            input_df = pd.read_csv(self.input_file)
            to_process = input_df
        return to_process

    def process(self):
        """Main method to process the DataFrame."""
        to_process = self._load_input_data()
        if to_process.empty:
            print("No rows to process. All rows are already processed.")
            return

        df = dd.from_pandas(to_process, npartitions=self.npartitions)

        with ProgressBar():
            processed_df = df.map_partitions(self._process_chunk).compute()

        # Save the processed rows to the output file
        if os.path.exists(self.output_file):
            output_df = pd.read_csv(self.output_file)
            output_df = pd.concat([output_df, processed_df])
        else:
            output_df = processed_df

        output_df.to_csv(self.output_file, index=False)

        # Save the updated storage
        self._save_storage()

# Example usage
if __name__ == "__main__":

    data_df = pd.DataFrame(
        {
            "input_column": [1, 2, 3, 4, 5],
            "key2": ["a", "b", "c", "d", "e"],
        }
    )

    data_df.to_csv("data.csv", index=False)
    processor = DataFrameProcessor(
        input_file="data.csv",
        output_file="output.csv",
        storage_file="persistent_storage.csv",
        api_url="https://api.example.com/data",
    )
    processor.process()
