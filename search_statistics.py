class SearchStatistics:
    def get_document_indexes_with_overlapped_scores(self, result_windowed_documents: list) -> dict[int, dict]:
        result_document_indexes_with_overlapped_scores = {}
        for windowed_document in result_windowed_documents:
            windowed_document_source_indexes = [windowed_document.meta["index_window"] + i for i in range(windowed_document.meta["window_size"])]
            
            for windowed_document_source_index in windowed_document_source_indexes:
                if (result_document_indexes_with_overlapped_scores.get(windowed_document_source_index, None) is None):
                    result_document_indexes_with_overlapped_scores[windowed_document_source_index] = {
                        "count": 1, 
                        "score_mean": windowed_document.score / 1
                    }
                else:
                    old_count = result_document_indexes_with_overlapped_scores[windowed_document_source_index]["count"]
                    new_count = old_count + 1
                    result_document_indexes_with_overlapped_scores[windowed_document_source_index]["count"] = new_count
                    new_value = windowed_document.score / 1
                    old_score_mean = result_document_indexes_with_overlapped_scores[windowed_document_source_index]["score_mean"]
                    new_score_mean = old_score_mean + ((new_value-old_score_mean)/new_count)
                    result_document_indexes_with_overlapped_scores[windowed_document_source_index]["score_mean"] = new_score_mean
                    
        return result_document_indexes_with_overlapped_scores
    
    
    def get_selected_labels(self, document_indexes_with_overlapped_scores: dict[int, dict], percentage: float) -> list:
        items = document_indexes_with_overlapped_scores.items()
        labels = []
        max_selection = int(percentage*len(items))
        count_selection = 0
        for index, stats in sorted(items, key=lambda item: item[1]["score_mean"], reverse=True):
            if count_selection >= max_selection:
                break
            score = f"{stats['score_mean']: .4f}"
            labels.append(score)
            count_selection += 1    
        
        return labels
    
search_statistics = SearchStatistics()