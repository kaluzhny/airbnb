import numpy as np
import pandas as pd
import math
from datetime import date
from sklearn.decomposition import PCA
from dataset import DataSet


def gender_idx(gender):
    if gender == 'FEMALE':
        return 1
    elif gender == 'MALE':
        return 2
    return 0


def str_to_date(str_date):
    y=int(str_date.split('-')[0])
    m=int(str_date.split('-')[1])
    d=int(str_date.split('-')[2])
    return date(y, m, d)


def date_diff(y1, m1, d1, y2, m2, d2):
    d1 = date(y1, m1, d1)
    d2 = date(y2, m2, d2)
    delta = d2 - d1
    return delta.days


def make_one_hot(df, feature, values=None, test_columns=None, use_threshold=False):
    dummy_df = pd.get_dummies(df[feature], prefix=feature)
    if values is not None:
        dummy_df = dummy_df[[feature + '_' + value for value in values]]
    return pd.concat((df, dummy_df), axis=1)


def do_pca(x):
    pca = PCA()
    pca.fit(x)
    print('variance_ratio: ', pca.explained_variance_ratio_)

    n_components = 0
    acc = 0.0
    for i, ratio in enumerate(pca.explained_variance_ratio_):
        acc += ratio
        if acc > 0.9999:
            n_components = i
            break

    pca = PCA(n_components=n_components)
    pca.fit(x)
    print('variance_ratio: ', pca.explained_variance_ratio_)
    return pca


def remove_sessions_columns(x):
    new_columns = [column for column in x.columns_ if not column.startswith('s_')]
    return x.filter_columns(new_columns)


def remove_no_sessions_columns(x, columns):
    new_columns_idx = []
    for i, column in enumerate(columns):
        if column.startswith('s_'):
            new_columns_idx.append(i)
    return x[:, new_columns_idx]


def divide_by_has_sessions(x, y):
    count_all_col_idx = list(x.columns_).index('year_first_active')
    x_ids = x.ids_
    x_data = x.data_

    rows_filter_sessions = (x_data[:, count_all_col_idx] == 2014)
    rows_filter_no_sessions = (x_data[:, count_all_col_idx] != 2014)

    x_ids_sessions = [x_ids[i] for i, f in enumerate(rows_filter_sessions) if f]
    x_data_sessions = x_data[rows_filter_sessions, :]
    x_sessions = DataSet(x_ids_sessions, x.columns_, x_data_sessions)

    x_ids_no_sessions = [x_ids[i] for i, f in enumerate(rows_filter_no_sessions) if f]
    x_data_no_sessions = x_data[rows_filter_no_sessions, :]
    x_no_sessions = DataSet(x_ids_no_sessions, x.columns_, x_data_no_sessions)

    if y is None:
        return x_sessions, x_no_sessions

    y_sessions = y[rows_filter_sessions, ]
    y_no_sessions = y[rows_filter_no_sessions, ]
    return x_sessions, y_sessions, x_no_sessions, y_no_sessions


def sync_columns(x_1, x_2):
    columns = list(set(x_1.columns_) & set(x_2.columns_))
    return x_1.filter_columns(columns), x_2.filter_columns(columns)


def sync_columns_2(x_1, x_2):
    columns = list(set(x_1.columns_) | set(x_2.columns_))
    return x_1.filter_columns(columns), x_2.filter_columns(columns)


def add_features(data_df):
    print('add_features <<')
    data_df = data_df.copy()

    data_df['age'] = data_df.apply(
        lambda r: 2015 - r['age'] if (np.isfinite(r['age']) & (r['age'] >= 1900) & (r['age'] < 2000)) else r['age'], axis=1)
    data_df['has_age'] = data_df.apply(lambda r: 0 if pd.isnull(r['age']) or r['age'] <= 16 or r['age'] >= 80 else 1, axis=1)

    data_df['age_imp'] = data_df.apply(lambda r: r['age'] if (np.isfinite(r['age']) & (r['age'] >= 16) & (r['age'] < 80)) else 0, axis=1)

    # mean_age = data_df[np.isfinite(data_df['age']) & (data_df['age'] < 100)]['age'].mean()
    # data_df['age_imp_mean'] = data_df.apply(
    #    lambda r: r['age'] if (np.isfinite(r['age']) & (r['age'] >= 16) & (r['age'] < 80)) else mean_age, axis=1)

    # data_df['day_account_created'] = data_df.apply(lambda r: int(str(r['date_account_created']).split('-')[2]), axis=1)
    # data_df['month_account_created'] = data_df.apply(lambda r: int(str(r['date_account_created']).split('-')[1]), axis=1)
    # data_df['year_account_created'] = data_df.apply(lambda r: int(str(r['date_account_created']).split('-')[0]), axis=1)
    # data_df['day_of_week'] = data_df.apply(lambda r: str_to_date(r['date_account_created']).weekday(), axis=1)
    # data_df['day_of_year'] = data_df.apply(lambda r: 12 * r['month_account_created'] + r['day_account_created'], axis=1)
    # data_df = data_df.drop(['day_account_created', 'month_account_created'], axis=1)

    data_df['day_first_active'] = data_df.apply(lambda r: int(str(r['timestamp_first_active'])[6:8]), axis=1)
    data_df['month_first_active'] = data_df.apply(lambda r: int(str(r['timestamp_first_active'])[4:6]), axis=1)
    data_df['year_first_active'] = data_df.apply(lambda r: int(str(r['timestamp_first_active'])[0:4]), axis=1)

    data_df['day_of_week_first_active'] = data_df.apply(
        lambda r: date(r['year_first_active'], r['month_first_active'], r['day_first_active']).weekday(), axis=1)

    data_df['day_of_year_first_active'] = data_df.apply(
        lambda r: 12 * r['month_first_active'] + r['day_first_active'], axis=1)
    # data_df = data_df.drop(['year_account_created', 'year_first_active'], axis=1)

    data_df = make_one_hot(data_df, 'language')
    data_df = make_one_hot(data_df, 'gender')
    data_df = make_one_hot(data_df, 'first_device_type')
    data_df = make_one_hot(data_df, 'affiliate_channel')
    data_df = make_one_hot(data_df, 'affiliate_provider')
    data_df = make_one_hot(data_df, 'first_affiliate_tracked')
    data_df = make_one_hot(data_df, 'signup_app')
    data_df = make_one_hot(data_df, 'signup_method')
    data_df = make_one_hot(data_df, 'first_browser')
    data_df = make_one_hot(data_df, 'signup_flow')

    drop_columns = ['date_account_created', 'timestamp_first_active', 'date_first_booking', 'gender', 'age',
                    'signup_method', 'language', 'affiliate_channel', 'affiliate_provider',
                    'first_affiliate_tracked', 'signup_app', 'first_device_type', 'first_browser' ]

    data_df = data_df.drop(drop_columns, axis=1)

    print('add_features >>')

    return data_df


def sessions_has_column_feature(data_df, sessions_df, column, feature):
    sessions_df=sessions_df[['user_id', column]]
    df_sessions_action_type = sessions_df[sessions_df[column] == feature]

    df_sessions_action_feature_counts = df_sessions_action_type.groupby(['user_id', column]).size().reset_index()
    df_sessions_action_feature_counts.columns = ['id', column, 'count']

    df_action_type_counts = pd.merge(data_df, df_sessions_action_feature_counts, on='id', how='left')[['id', 'count',
                                                                                                       's_count_all']]

    df_action_type_counts['count'] = df_action_type_counts.apply(
        lambda r : r['count'] if not math.isnan(r['count']) else (0 if r['s_count_all'] == 0 else 0), axis=1)

    return df_action_type_counts['count']


def session_unique_devices_count(data_df, sessions_df):
    df_sessions_devices = sessions_df[['user_id', 'device_type']]
    df_sessions_devices = df_sessions_devices.groupby(['user_id'])['device_type'].nunique().reset_index()
    df_sessions_devices.columns=['id', 'count']
    df_sessions_devices_counts = pd.merge(data_df, df_sessions_devices, on='id', how='left')[[
        'id', 'count', 's_count_all']]
    df_sessions_devices_counts['count'] = df_sessions_devices_counts.apply(
        lambda r : r['count'] if not math.isnan(r['count']) else (0 if r['s_count_all'] == 0 else 0), axis=1)
    return df_sessions_devices_counts['count']


def sessions_has_action_detail(data_df, sessions_df, action_detail):
    return sessions_has_column_feature(data_df, sessions_df, 'action_detail', action_detail)


def sessions_has_action_type(data_df, sessions_df, action_type):
    return sessions_has_column_feature(data_df, sessions_df, 'action_type', action_type)


def sessions_has_action(data_df, sessions_df, action):
    return sessions_has_column_feature(data_df, sessions_df, 'action', action)


def add_sessions_features(data_df, sessions_df):

    print('add_sessions_data <<')

    sessions_actions_count_df=pd.DataFrame({'s_count_all' : sessions_df.groupby(['user_id']).size()}).reset_index()
    sessions_actions_count_df.columns=['id', 's_count_all']
    data_df = pd.merge(data_df, sessions_actions_count_df, on='id', how='left')
    data_df['s_count_all'] = data_df.apply(lambda r: math.log(r['s_count_all'] + 1) if (r['s_count_all'] > 0) else 0, axis=1)
    # return data_df

    df_sessions_secs = sessions_df[['user_id', 'secs_elapsed']]
    df_sessions_secs = df_sessions_secs.groupby(['user_id']).sum().reset_index()
    df_sessions_secs.columns=['id', 's_secs_elapsed']
    data_df = pd.merge(data_df, df_sessions_secs, on='id', how='left')
    data_df['s_secs_elapsed'] = data_df.apply(lambda r: math.log(r['s_secs_elapsed'] + 1) if (r['s_secs_elapsed'] > 0) else 0, axis=1)

    data_df['s_unique_devices'] = session_unique_devices_count(data_df, sessions_df)

    # for action in sessions_df['action'].unique():
    #     if isinstance(action, str):
    #         data_df['s_has_action_' + action] = sessions_has_action(data_df, sessions_df, action)

    data_df['s_has_action_show'] = sessions_has_action(data_df, sessions_df, 'show')
    data_df['s_has_action_index'] = sessions_has_action(data_df, sessions_df, 'index')
    data_df['s_has_action_search_results'] = sessions_has_action(data_df, sessions_df, 'search_results')
    data_df['s_has_action_personalize'] = sessions_has_action(data_df, sessions_df, 'personalize')
    data_df['s_has_action_search'] = sessions_has_action(data_df, sessions_df, 'search')
    data_df['s_has_action_ajax_refresh_subtotal'] = sessions_has_action(data_df, sessions_df, 'ajax_refresh_subtotal')
    data_df['s_has_action_update'] = sessions_has_action(data_df, sessions_df, 'update')
    data_df['s_has_action_similar_listings'] = sessions_has_action(data_df, sessions_df, 'similar_listings')
    data_df['s_has_action_social_connections'] = sessions_has_action(data_df, sessions_df, 'social_connections')
    data_df['s_has_action_reviews'] = sessions_has_action(data_df, sessions_df, 'reviews')
    data_df['s_has_action_active'] = sessions_has_action(data_df, sessions_df, 'active')
    data_df['s_has_action_similar_listings_v2'] = sessions_has_action(data_df, sessions_df, 'similar_listings_v2')
    data_df['s_has_action_lookup'] = sessions_has_action(data_df, sessions_df, 'lookup')
    data_df['s_has_action_create'] = sessions_has_action(data_df, sessions_df, 'create')
    data_df['s_has_action_dashboard'] = sessions_has_action(data_df, sessions_df, 'dashboard')
    data_df['s_has_action_header_userpic'] = sessions_has_action(data_df, sessions_df, 'header_userpic')
    data_df['s_has_action_collections'] = sessions_has_action(data_df, sessions_df, 'collections')
    data_df['s_has_action_edit'] = sessions_has_action(data_df, sessions_df, 'edit')
    data_df['s_has_action_campaigns'] = sessions_has_action(data_df, sessions_df, 'campaigns')
    data_df['s_has_action_track_page_view'] = sessions_has_action(data_df, sessions_df, 'track_page_view')
    data_df['s_has_action_unavailabilities'] = sessions_has_action(data_df, sessions_df, 'unavailabilities')
    data_df['s_has_action_qt2'] = sessions_has_action(data_df, sessions_df, 'qt2')
    data_df['s_has_action_notifications'] = sessions_has_action(data_df, sessions_df, 'notifications')
    data_df['s_has_action_confirm_email'] = sessions_has_action(data_df, sessions_df, 'confirm_email')
    data_df['s_has_action_requested'] = sessions_has_action(data_df, sessions_df, 'requested')
    data_df['s_has_action_identity'] = sessions_has_action(data_df, sessions_df, 'identity')
    data_df['s_has_action_ajax_check_dates'] = sessions_has_action(data_df, sessions_df, 'ajax_check_dates')
    data_df['s_has_action_show_personalize'] = sessions_has_action(data_df, sessions_df, 'show_personalize')
    data_df['s_has_action_ask_question'] = sessions_has_action(data_df, sessions_df, 'ask_question')
    data_df['s_has_action_listings'] = sessions_has_action(data_df, sessions_df, 'listings')
    data_df['s_has_action_authenticate'] = sessions_has_action(data_df, sessions_df, 'authenticate')
    data_df['s_has_action_calendar_tab_inner2 '] = sessions_has_action(data_df, sessions_df, 'calendar_tab_inner2')
    data_df['s_has_action_travel_plans_current'] = sessions_has_action(data_df, sessions_df, 'travel_plans_current')
    data_df['s_has_action_edit_verification'] = sessions_has_action(data_df, sessions_df, 'edit_verification')
    data_df['s_has_action_ajax_lwlb_contact'] = sessions_has_action(data_df, sessions_df, 'ajax_lwlb_contact')
    data_df['s_has_action_other_hosting_reviews_first'] = sessions_has_action(data_df, sessions_df, 'other_hosting_reviews_first')
    data_df['s_has_action_recommendations'] = sessions_has_action(data_df, sessions_df, 'recommendations')
    data_df['s_has_action_manage_listing'] = sessions_has_action(data_df, sessions_df, 'manage_listing')
    data_df['s_has_action_click'] = sessions_has_action(data_df, sessions_df, 'click')
    data_df['s_has_action_complete_status'] = sessions_has_action(data_df, sessions_df, 'complete_status')
    data_df['s_has_action_ajax_photo_widget_form_iframe'] = sessions_has_action(data_df, sessions_df, 'ajax_photo_widget_form_iframe')
    data_df['s_has_action_payment_instruments'] = sessions_has_action(data_df, sessions_df, 'payment_instruments')
    data_df['s_has_action_message_to_host_focus'] = sessions_has_action(data_df, sessions_df, 'message_to_host_focus')
    data_df['s_has_action_verify'] = sessions_has_action(data_df, sessions_df, 'verify')
    data_df['s_has_action_payment_methods'] = sessions_has_action(data_df, sessions_df, 'payment_methods')
    data_df['s_has_action_cancellation_policies'] = sessions_has_action(data_df, sessions_df, 'cancellation_policies')
    data_df['s_has_action_callback'] = sessions_has_action(data_df, sessions_df, 'callback')
    data_df['s_has_action_settings'] = sessions_has_action(data_df, sessions_df, 'settings')
    data_df['s_has_action_custom_recommended_destinations'] = sessions_has_action(data_df, sessions_df, 'custom_recommended_destinations')
    data_df['s_has_action_pending'] = sessions_has_action(data_df, sessions_df, 'pending')
    data_df['s_has_action_profile_pic'] = sessions_has_action(data_df, sessions_df, 'profile_pic')
    data_df['s_has_action_populate_help_dropdown'] = sessions_has_action(data_df, sessions_df, 'populate_help_dropdown')
    data_df['s_has_action_message_to_host_change'] = sessions_has_action(data_df, sessions_df, 'message_to_host_change')
    data_df['s_has_action_ajax_image_upload'] = sessions_has_action(data_df, sessions_df, 'ajax_image_upload')
    data_df['s_has_action_view'] = sessions_has_action(data_df, sessions_df, 'view')
    data_df['s_has_action_kba_update'] = sessions_has_action(data_df, sessions_df, 'kba_update')
    data_df['s_has_action_references'] = sessions_has_action(data_df, sessions_df, 'references')
    data_df['s_has_action_my'] = sessions_has_action(data_df, sessions_df, 'my')
    data_df['s_has_action_ajax_get_referrals_amt'] = sessions_has_action(data_df, sessions_df, 'ajax_get_referrals_amt')
    data_df['s_has_action_new'] = sessions_has_action(data_df, sessions_df, 'new')
    data_df['s_has_action_agree_terms_check'] = sessions_has_action(data_df, sessions_df, 'agree_terms_check')
    data_df['s_has_action_apply_reservation'] = sessions_has_action(data_df, sessions_df, 'apply_reservation')
    data_df['s_has_action_connect'] = sessions_has_action(data_df, sessions_df, 'connect')
    data_df['s_has_action_recommended_listings'] = sessions_has_action(data_df, sessions_df, 'recommended_listings')
    data_df['s_has_action_faq'] = sessions_has_action(data_df, sessions_df, 'faq')
    data_df['s_has_action_populate_from_facebook'] = sessions_has_action(data_df, sessions_df, 'populate_from_facebook')
    data_df['s_has_action_account'] = sessions_has_action(data_df, sessions_df, 'account')
    data_df['s_has_action_available'] = sessions_has_action(data_df, sessions_df, 'available')
    data_df['s_has_action_jumio_token'] = sessions_has_action(data_df, sessions_df, 'jumio_token')
    data_df['s_has_action_qt_reply_v2'] = sessions_has_action(data_df, sessions_df, 'qt_reply_v2')
    data_df['s_has_action_signup_login'] = sessions_has_action(data_df, sessions_df, 'signup_login')
    data_df['s_has_action_request_new_confirm_email'] = sessions_has_action(data_df, sessions_df, 'request_new_confirm_email')
    data_df['s_has_action_kba'] = sessions_has_action(data_df, sessions_df, 'kba')
    data_df['s_has_action_handle_vanity_url'] = sessions_has_action(data_df, sessions_df, 'handle_vanity_url')
    data_df['s_has_action_coupon_field_focus'] = sessions_has_action(data_df, sessions_df, 'coupon_field_focus')
    data_df['s_has_action_phone_number_widget'] = sessions_has_action(data_df, sessions_df, 'phone_number_widget')
    data_df['s_has_action_open_graph_setting'] = sessions_has_action(data_df, sessions_df, 'open_graph_setting')
    data_df['s_has_action_set_user'] = sessions_has_action(data_df, sessions_df, 'set_user')
    data_df['s_has_action_faq_category'] = sessions_has_action(data_df, sessions_df, 'faq_category')
    data_df['s_has_action_apply_coupon_click'] = sessions_has_action(data_df, sessions_df, 'apply_coupon_click')
    data_df['s_has_action_reviews_new'] = sessions_has_action(data_df, sessions_df, 'reviews_new')
    data_df['s_has_action_apply_coupon_error'] = sessions_has_action(data_df, sessions_df, 'apply_coupon_error')
    data_df['s_has_action_apply_coupon_error_type'] = sessions_has_action(data_df, sessions_df, 'apply_coupon_error_type')
    data_df['s_has_action_localization_settings'] = sessions_has_action(data_df, sessions_df, 'localization_settings')
    data_df['s_has_action_languages_multiselect'] = sessions_has_action(data_df, sessions_df, 'languages_multiselect')
    data_df['s_has_action_at_checkpoint'] = sessions_has_action(data_df, sessions_df, 'at_checkpoint')
    data_df['s_has_action_jumio_redirect'] = sessions_has_action(data_df, sessions_df, 'jumio_redirect')
    data_df['s_has_action_delete'] = sessions_has_action(data_df, sessions_df, 'delete')
    data_df['s_has_action_ajax_referral_banner_experiment_type'] = sessions_has_action(data_df, sessions_df, 'ajax_referral_banner_experiment_type')
    data_df['s_has_action_login'] = sessions_has_action(data_df, sessions_df, 'login')
    data_df['s_has_action_endpoint_error'] = sessions_has_action(data_df, sessions_df, 'endpoint_error')
    data_df['s_has_action_payout_preferences'] = sessions_has_action(data_df, sessions_df, 'payout_preferences')
    data_df['s_has_action_complete_redirect'] = sessions_has_action(data_df, sessions_df, 'complete_redirect')
    data_df['s_has_action_status'] = sessions_has_action(data_df, sessions_df, 'status')
    data_df['s_has_action_ajax_referral_banner_type'] = sessions_has_action(data_df, sessions_df, 'ajax_referral_banner_type')
    data_df['s_has_action_hosting_social_proof'] = sessions_has_action(data_df, sessions_df, 'hosting_social_proof')
    data_df['s_has_action_referrer_status'] = sessions_has_action(data_df, sessions_df, 'referrer_status')
    data_df['s_has_action_facebook_auto_login'] = sessions_has_action(data_df, sessions_df, 'facebook_auto_login')
    data_df['s_has_action_read_policy_click'] = sessions_has_action(data_df, sessions_df, 'read_policy_click')
    data_df['s_has_action_this_hosting_reviews'] = sessions_has_action(data_df, sessions_df, 'this_hosting_reviews')
    data_df['s_has_action_cancellation_policy_click'] = sessions_has_action(data_df, sessions_df, 'cancellation_policy_click')
    data_df['s_has_action_uptodate'] = sessions_has_action(data_df, sessions_df, 'uptodate')
    data_df['s_has_action_push_notification_callback'] = sessions_has_action(data_df, sessions_df, 'push_notification_callback')
    data_df['s_has_action_tell_a_friend'] = sessions_has_action(data_df, sessions_df, 'tell_a_friend')
    data_df['s_has_action_10'] = sessions_has_action(data_df, sessions_df, '10')
    data_df['s_has_action_phone_verification_number_submitted_for_sms'] = sessions_has_action(data_df, sessions_df, 'phone_verification_number_submitted_for_sms')
    data_df['s_has_action_tos_confirm'] = sessions_has_action(data_df, sessions_df, 'tos_confirm')
    data_df['s_has_action_coupon_code_click'] = sessions_has_action(data_df, sessions_df, 'coupon_code_click')
    data_df['s_has_action_decision_tree'] = sessions_has_action(data_df, sessions_df, 'decision_tree')
    data_df['s_has_action_recent_reservations'] = sessions_has_action(data_df, sessions_df, 'recent_reservations')
    data_df['s_has_action_phone_verification_number_sucessfully_submitted'] = sessions_has_action(data_df, sessions_df, 'phone_verification_number_sucessfully_submitted')
    data_df['s_has_action_pay'] = sessions_has_action(data_df, sessions_df, 'pay')
    data_df['s_has_action_12'] = sessions_has_action(data_df, sessions_df, '12')
    data_df['s_has_action_popular'] = sessions_has_action(data_df, sessions_df, 'popular')
    data_df['s_has_action_host_summary'] = sessions_has_action(data_df, sessions_df, 'host_summary')
    data_df['s_has_action_create_multiple'] = sessions_has_action(data_df, sessions_df, 'create_multiple')
    data_df['s_has_action_transaction_history'] = sessions_has_action(data_df, sessions_df, 'transaction_history')
    data_df['s_has_action_phone_verification_success'] = sessions_has_action(data_df, sessions_df, 'phone_verification_success')
    data_df['s_has_action_login_modal'] = sessions_has_action(data_df, sessions_df, 'login_modal')

    data_df['s_has_view'] = sessions_has_action_type(data_df, sessions_df, 'view')
    data_df['s_has_data'] = sessions_has_action_type(data_df, sessions_df, 'data')
    data_df['s_has_click'] = sessions_has_action_type(data_df, sessions_df, 'click')
    data_df['s_has_unknown'] = sessions_has_action_type(data_df, sessions_df, '-unknown-')
    data_df['s_has_message_post'] = sessions_has_action_type(data_df, sessions_df, 'message_post')
    data_df['s_has_submit'] = sessions_has_action_type(data_df, sessions_df, 'submit')
    data_df['s_has_booking_request'] = sessions_has_action_type(data_df, sessions_df, 'booking_request')
    data_df['s_has_modify'] = sessions_has_action_type(data_df, sessions_df, 'modify')
    data_df['s_has_partner_callback'] = sessions_has_action_type(data_df, sessions_df, 'partner_callback')

    data_df['s_has_action_detail_view_search_results'] = sessions_has_action_detail(data_df, sessions_df, 'view_search_results')
    data_df['s_has_action_detail_p3'] = sessions_has_action_detail(data_df, sessions_df, 'p3')
    data_df['s_has_action_detail_unknown'] = sessions_has_action_detail(data_df, sessions_df, '-unknown-')
    data_df['s_has_action_detail_wishlist_content_update'] = sessions_has_action_detail(data_df, sessions_df, 'wishlist_content_update')
    data_df['s_has_action_detail_user_profile'] = sessions_has_action_detail(data_df, sessions_df, 'user_profile')
    data_df['s_has_action_detail_change_trip_characteristics'] = sessions_has_action_detail(data_df, sessions_df, 'change_trip_characteristics')
    data_df['s_has_action_detail_similar_listings'] = sessions_has_action_detail(data_df, sessions_df, 'similar_listings')
    data_df['s_has_action_detail_update_listing'] = sessions_has_action_detail(data_df, sessions_df, 'update_listing')
    data_df['s_has_action_detail_listing_reviews'] = sessions_has_action_detail(data_df, sessions_df, 'listing_reviews')
    data_df['s_has_action_detail_dashboard'] = sessions_has_action_detail(data_df, sessions_df, 'dashboard')
    data_df['s_has_action_detail_user_wishlists'] = sessions_has_action_detail(data_df, sessions_df, 'user_wishlists')
    data_df['s_has_action_detail_header_userpic'] = sessions_has_action_detail(data_df, sessions_df, 'header_userpic')
    data_df['s_has_action_detail_message_thread'] = sessions_has_action_detail(data_df, sessions_df, 'message_thread')
    data_df['s_has_action_detail_edit_profile'] = sessions_has_action_detail(data_df, sessions_df, 'edit_profile')
    data_df['s_has_action_detail_message_post'] = sessions_has_action_detail(data_df, sessions_df, 'message_post')
    data_df['s_has_action_detail_contact_host'] = sessions_has_action_detail(data_df, sessions_df, 'contact_host')
    data_df['s_has_action_detail_unavailable_dates'] = sessions_has_action_detail(data_df, sessions_df, 'unavailable_dates')
    data_df['s_has_action_detail_confirm_email_link'] = sessions_has_action_detail(data_df, sessions_df, 'confirm_email_link')
    data_df['s_has_action_detail_create_user'] = sessions_has_action_detail(data_df, sessions_df, 'create_user')
    data_df['s_has_action_detail_change_contact_host_dates'] = sessions_has_action_detail(data_df, sessions_df, 'change_contact_host_dates')
    data_df['s_has_action_detail_user_profile_content_update'] = sessions_has_action_detail(data_df, sessions_df, 'user_profile_content_update')
    data_df['s_has_action_detail_user_reviews'] = sessions_has_action_detail(data_df, sessions_df, 'user_reviews')
    data_df['s_has_action_detail_p5'] = sessions_has_action_detail(data_df, sessions_df, 'p5')
    data_df['s_has_action_detail_login'] = sessions_has_action_detail(data_df, sessions_df, 'login')
    data_df['s_has_action_detail_your_trips'] = sessions_has_action_detail(data_df, sessions_df, 'your_trips')
    data_df['s_has_action_detail_p1'] = sessions_has_action_detail(data_df, sessions_df, 'p1')
    data_df['s_has_action_detail_notifications'] = sessions_has_action_detail(data_df, sessions_df, 'notifications')
    data_df['s_has_action_detail_profile_verifications'] = sessions_has_action_detail(data_df, sessions_df, 'profile_verifications')
    data_df['s_has_action_detail_reservations'] = sessions_has_action_detail(data_df, sessions_df, 'reservations')
    data_df['s_has_action_detail_user_listings'] = sessions_has_action_detail(data_df, sessions_df, 'user_listings')
    data_df['s_has_action_detail_your_listings'] = sessions_has_action_detail(data_df, sessions_df, 'your_listings')
    data_df['s_has_action_detail_listing_recommendations'] = sessions_has_action_detail(data_df, sessions_df, 'listing_recommendations')
    data_df['s_has_action_detail_update_user'] = sessions_has_action_detail(data_df, sessions_df, 'update_user')
    data_df['s_has_action_detail_create_phone_numbers'] = sessions_has_action_detail(data_df, sessions_df, 'create_phone_numbers')
    data_df['s_has_action_detail_p4'] = sessions_has_action_detail(data_df, sessions_df, 'p4')
    data_df['s_has_action_detail_update_listing_description'] = sessions_has_action_detail(data_df, sessions_df, 'update_listing_description')
    data_df['s_has_action_detail_update_user_profile'] = sessions_has_action_detail(data_df, sessions_df, 'update_user_profile')
    data_df['s_has_action_detail_manage_listing'] = sessions_has_action_detail(data_df, sessions_df, 'manage_listing')
    data_df['s_has_action_detail_payment_instruments'] = sessions_has_action_detail(data_df, sessions_df, 'payment_instruments')
    data_df['s_has_action_detail_account_notification_settings'] = sessions_has_action_detail(data_df, sessions_df, 'account_notification_settings')
    data_df['s_has_action_detail_message_to_host_focus'] = sessions_has_action_detail(data_df, sessions_df, 'message_to_host_focus')
    data_df['s_has_action_detail_signup'] = sessions_has_action_detail(data_df, sessions_df, 'signup')
    data_df['s_has_action_detail_cancellation_policies'] = sessions_has_action_detail(data_df, sessions_df, 'cancellation_policies')
    data_df['s_has_action_detail_oauth_response'] = sessions_has_action_detail(data_df, sessions_df, 'oauth_response')
    data_df['s_has_action_detail_message_inbox'] = sessions_has_action_detail(data_df, sessions_df, 'message_inbox')
    data_df['s_has_action_detail_view_listing'] = sessions_has_action_detail(data_df, sessions_df, 'view_listing')
    data_df['s_has_action_detail_message_to_host_change'] = sessions_has_action_detail(data_df, sessions_df, 'message_to_host_change')
    data_df['s_has_action_detail_list_your_space'] = sessions_has_action_detail(data_df, sessions_df, 'list_your_space')
    data_df['s_has_action_detail_pending'] = sessions_has_action_detail(data_df, sessions_df, 'pending')
    data_df['s_has_action_detail_wishlist'] = sessions_has_action_detail(data_df, sessions_df, 'wishlist')
    data_df['s_has_action_detail_profile_references'] = sessions_has_action_detail(data_df, sessions_df, 'profile_references')
    data_df['s_has_action_detail_apply_coupon'] = sessions_has_action_detail(data_df, sessions_df, 'apply_coupon')
    data_df['s_has_action_detail_oauth_login'] = sessions_has_action_detail(data_df, sessions_df, 'oauth_login')
    data_df['s_has_action_detail_view_reservations'] = sessions_has_action_detail(data_df, sessions_df, 'view_reservations')
    data_df['s_has_action_detail_login_page'] = sessions_has_action_detail(data_df, sessions_df, 'login_page')
    data_df['s_has_action_detail_post_checkout_action'] = sessions_has_action_detail(data_df, sessions_df, 'post_checkout_action')
    data_df['s_has_action_detail_trip_availability'] = sessions_has_action_detail(data_df, sessions_df, 'trip_availability')
    data_df['s_has_action_detail_send_message'] = sessions_has_action_detail(data_df, sessions_df, 'send_message')
    data_df['s_has_action_detail_signup_login_page'] = sessions_has_action_detail(data_df, sessions_df, 'signup_login_page')


    print('add_sessions_data >>')

    return data_df




def print_columns(columns):
    print('columns: ', ['f'+str(idx)+': '+col for idx, col in enumerate(columns)])