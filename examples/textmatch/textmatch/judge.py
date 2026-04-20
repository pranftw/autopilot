class RuleJudge:
  def categorize_failure(self, datum):
    meta = datum.metadata
    failure_type = meta.get('failure_type', 'unknown')
    return {
      'category': failure_type,
      'item_id': datum.item_id,
      'details': {
        'predicted': meta.get('predicted', ''),
        'expected': meta.get('expected', ''),
      },
    }

  def run(self, results):
    failures = [r for r in results if not r.success]
    categorized = [self.categorize_failure(f) for f in failures]
    summary = {}
    for c in categorized:
      cat = c['category']
      summary[cat] = summary.get(cat, 0) + 1
    return {
      'failures': categorized,
      'summary': summary,
      'total_failures': len(failures),
    }
