# html_reporter.py
from jinja2 import Environment, FileSystemLoader
import os
# import base64 # For embedding images directly if preferred, or use file paths

# def fig_to_base64(fig_path):
#     """Converts an image file to a base64 string for embedding in HTML."""
#     if not os.path.exists(fig_path):
#         return None
#     with open(fig_path, "rb") as image_file:
#         encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
#     return f"data:image/png;base64,{encoded_string}"

def generate_html_report(report_data, template_name="report_template.html", output_filename="assignment_report.html"):
    """
    Generates an HTML report from the collected data.
    """
    if not os.path.exists("plots"): # Should have been created by main
        os.makedirs("plots")
        
    # Ensure relative paths for images if they exist
    for part_key, part_results in report_data.items():
        if isinstance(part_results, dict):
            for key, value in part_results.items():
                if isinstance(value, str) and value.startswith("plots" + os.sep) and value.endswith(".png"):
                    # Make path relative if it's absolute for some reason, or just ensure it's correct
                    report_data[part_key][key] = os.path.join(value) # os.path.join will normalize
                elif key == "confusion_matrix_array" and value is not None: # Ensure it's a list for Jinja
                     report_data[part_key][key] = list(value)


    env = Environment(loader=FileSystemLoader('.')) 
    try:
        template = env.get_template(template_name)
    except Exception as e:
        print(f"Error loading template {template_name}: {e}")
        print("Make sure 'report_template.html' exists in the current directory.")
        return

    html_content = template.render(data=report_data)

    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"HTML report generated: {output_filename}")